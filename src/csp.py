"""optimal-smugglers-cove CSP/ILP solver.

Implements the pipeline described in `SOLVER_PROMPT.md`:
- Load + normalize ingredients (handle `Rum Category`)
- Build cocktail -> ingredient sets
- Join scores
- Solve the ILP with OR-Tools CP-SAT
- Emit required intermediate artifacts for insight/debug
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from ortools.sat.python import cp_model


@dataclass(frozen=True)
class Cocktail:
    name: str
    ingredients: tuple[str, ...]
    base_score: float
    weight: float


DEFAULT_CONFIG: dict[str, Any] = {
    "max_bottles": 10,
    "alpha": 0.3,
    "use_log_scores": True,
    "score_scale": 1.0,
    # Not in SOLVER_PROMPT.md, but required as an option.
    "use_rank_scores": False,
    # If true, select non-bottles from metadata.csv (e.g. lime, syrups).
    # Otherwise select bottles only.
    "use_ingredients": False,
    # CP-SAT requires integer coefficients.
    "objective_scale": 1_000_000,
}


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return dict(DEFAULT_CONFIG)

    loaded = json.loads(config_path.read_text(encoding="utf-8"))
    merged = dict(DEFAULT_CONFIG)
    merged.update(loaded)
    return merged


def load_bottle_status_map(metadata_csv: Path) -> dict[str, bool]:
    """ingredient -> is_bottle (from metadata.csv)."""

    bottle_status: dict[str, bool] = {}
    with metadata_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ing = (row.get("ingredient") or "").strip()
            if not ing:
                continue

            bottle_status[ing] = (row.get("bottle") or "").strip().upper() == "TRUE"
    return bottle_status


def normalize_ingredient(row: dict[str, str]) -> str:
    # `Rum Category` is a numeric id in this dataset; the canonical (normalized)
    # ingredient name is already in the `Ingredient` column.
    rum_cat = (row.get("Rum Category") or "").strip()
    _ = rum_cat  # kept for readability of the intended logic
    return (row.get("Ingredient") or "").strip()


def load_cocktails(
    index_csv: Path,
    bottle_status: dict[str, bool],
    *,
    include_bottles: bool,
) -> dict[str, set[str]]:
    cocktails: dict[str, set[str]] = defaultdict(set)

    with index_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            drink = (row.get("Drink Name") or "").strip()
            if not drink:
                continue

            ing = normalize_ingredient(row)
            if not ing:
                continue

            # Use metadata.csv to decide what counts as an item.
            # If the ingredient is missing from metadata.csv, keep it (robustness).
            is_bottle = bottle_status.get(ing)
            if is_bottle is None:
                cocktails[drink].add(ing)
                continue

            if include_bottles and is_bottle:
                cocktails[drink].add(ing)
            elif (not include_bottles) and (not is_bottle):
                cocktails[drink].add(ing)

    return {drink: set(ings) for drink, ings in cocktails.items() if ings}


def load_scores(scores_csv: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    with scores_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            drink = (row.get("drink") or "").strip()
            if not drink:
                continue
            raw_score = (row.get("score") or "").strip()
            try:
                scores[drink] = float(raw_score)
            except ValueError:
                pass
    return scores


def compute_cocktails(
    cocktails_raw: dict[str, set[str]],
    scores: dict[str, float],
    *,
    default_score: float,
    use_rank_scores: bool,
    use_log_scores: bool,
    score_scale: float,
) -> list[Cocktail]:
    cocktail_names = sorted(cocktails_raw.keys())

    base_by_drink: dict[str, float] = {
        name: float(scores.get(name, default_score)) for name in cocktail_names
    }

    if use_rank_scores:
        # Higher score = better (rank 1 best). Break ties deterministically by name.
        sorted_by_score = sorted(cocktail_names, key=lambda n: (-base_by_drink[n], n))
        base_by_drink = {name: float(i + 1) for i, name in enumerate(sorted_by_score)}

    def compute_weight(base_score: float) -> float:
        scaled = base_score * score_scale
        if use_log_scores:
            return math.log(1.0 + scaled)
        return scaled

    cocktails: list[Cocktail] = []
    for drink in cocktail_names:
        ings_sorted = tuple(sorted(cocktails_raw[drink]))
        base_score = base_by_drink[drink]
        cocktails.append(
            Cocktail(
                name=drink,
                ingredients=ings_sorted,
                base_score=base_score,
                weight=compute_weight(base_score),
            )
        )
    return cocktails


def solve(
    *,
    cocktails: list[Cocktail],
    max_bottles: int,
    alpha: float,
    objective_scale: int,
) -> tuple[list[str], list[str], float, dict[str, int]]:
    ingredients: set[str] = set()
    for c in cocktails:
        ingredients.update(c.ingredients)
    ingredient_list = sorted(ingredients)

    model = cp_model.CpModel()

    ingredient_vars = {ing: model.NewBoolVar(f"x::{ing}") for ing in ingredient_list}
    cocktail_vars = {c.name: model.NewBoolVar(f"y::{c.name}") for c in cocktails}

    model.Add(sum(ingredient_vars.values()) <= max_bottles)

    # No partial cocktails: y_c => all required x_i.
    for c in cocktails:
        y = cocktail_vars[c.name]
        k = len(c.ingredients)
        sum_x = sum(ingredient_vars[ing] for ing in c.ingredients)
        for ing in c.ingredients:
            model.Add(y <= ingredient_vars[ing])

        # Reverse implication: if all x_i are 1 then y_c must be 1.
        # y_c >= sum(x_i) - (k-1)
        model.Add(y >= sum_x - (k - 1))

    # Objective: maximize alpha*#cocktails + (1-alpha)*sum(weighted)
    # => sum( (alpha + (1-alpha)*weight) * y_c )
    scaled_terms: list[tuple[int, cp_model.IntVar]] = []
    for c in cocktails:
        coeff = alpha + (1.0 - alpha) * c.weight
        scaled = int(round(coeff * objective_scale))
        if scaled != 0:
            scaled_terms.append((scaled, cocktail_vars[c.name]))
    model.Maximize(sum(scaled * var for scaled, var in scaled_terms))

    solver = cp_model.CpSolver()
    solver.parameters.random_seed = 0
    solver.parameters.num_search_workers = 1

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"No solution found (status={status}).")

    selected_ingredients = [
        ing for ing, var in ingredient_vars.items() if solver.Value(var) == 1
    ]
    enabled_cocktails = [
        c.name for c in cocktails if solver.Value(cocktail_vars[c.name]) == 1
    ]
    enabled_set = set(enabled_cocktails)

    enabled_weight = sum(c.weight for c in cocktails if c.name in enabled_set)

    ingredient_marginals: dict[str, int] = {ing: 0 for ing in ingredient_list}
    for c in cocktails:
        if c.name not in enabled_set:
            continue
        for ing in c.ingredients:
            ingredient_marginals[ing] += 1

    selected_ingredients.sort()
    enabled_cocktails.sort()
    return selected_ingredients, enabled_cocktails, enabled_weight, ingredient_marginals


def compute_coverage_progression(
    *,
    cocktails: list[Cocktail],
    selected_ingredients: list[str],
    ingredient_frequency: dict[str, int],
) -> list[dict[str, Any]]:
    selected_set: set[str] = set()
    steps: list[dict[str, Any]] = []

    ordered = sorted(
        selected_ingredients,
        key=lambda ing: (-ingredient_frequency.get(ing, 0), ing),
    )

    for ing in ordered:
        selected_set.add(ing)
        unlocked = [
            c.name for c in cocktails if set(c.ingredients).issubset(selected_set)
        ]
        unlocked.sort()
        steps.append(
            {
                "added_ingredient": ing,
                "unlocked_cocktails": unlocked,
                "total_unlocked": len(unlocked),
            }
        )
    return steps


def write_intermediate_artifacts(
    *,
    output_dir: Path,
    cocktails: list[Cocktail],
    ingredient_set: list[str],
    ingredient_to_cocktails: dict[str, list[str]],
    cooccurrence_counts: dict[tuple[str, str], int],
    selected_ingredients: list[str],
    enabled_cocktails: list[str],
    ingredient_marginals: dict[str, int],
    coverage_progression: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ingredients.csv
    with (output_dir / "ingredients.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ingredient", "frequency"])
        rows = [
            (ing, len(ingredient_to_cocktails.get(ing, []))) for ing in ingredient_set
        ]
        # Sort by frequency descending; tie-break by ingredient name.
        rows.sort(key=lambda x: (-x[1], x[0]))
        for ing, freq in rows:
            writer.writerow([ing, freq])

    # cocktails_expanded.csv
    with (output_dir / "cocktails_expanded.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["drink", "ingredient_list", "score", "weight"])
        for c in sorted(cocktails, key=lambda x: x.name):
            writer.writerow(
                [c.name, json.dumps(list(c.ingredients)), c.base_score, c.weight]
            )

    # ingredient_cooccurrence.csv
    with (output_dir / "ingredient_cooccurrence.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["ingredient_a", "ingredient_b", "count"])
        rows = [
            (a, b, cooccurrence_counts[(a, b)]) for (a, b) in cooccurrence_counts.keys()
        ]
        # Sort by cooccurrence count descending; then deterministically.
        rows.sort(key=lambda x: (-x[2], x[0], x[1]))
        for a, b, count in rows:
            writer.writerow([a, b, count])

    # solution_marginals.csv
    with (output_dir / "solution_marginals.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["ingredient", "cocktails_enabled_in_solution"])
        for ing in sorted(ingredient_set):
            writer.writerow([ing, ingredient_marginals.get(ing, 0)])

    # coverage_progression.json
    payload = {
        "selected_ingredients": selected_ingredients,
        "enabled_cocktails": enabled_cocktails,
        "coverage_progression": coverage_progression,
    }
    (output_dir / "coverage_progression.json").write_text(
        json.dumps(payload, indent=2, sort_keys=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="index.csv", help="Path to index.csv")
    parser.add_argument("--scores", default="scores.csv", help="Path to scores.csv")
    parser.add_argument(
        "--metadata",
        default="metadata.csv",
        help="Path to metadata.csv (for bottle filtering)",
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to config.json (optional; defaults from SOLVER_PROMPT.md)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write intermediate artifacts",
    )
    args = parser.parse_args()

    index_csv = Path(args.index)
    scores_csv = Path(args.scores)
    metadata_csv = Path(args.metadata)
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)

    cfg = load_config(config_path)
    max_bottles = int(cfg["max_bottles"])
    alpha = float(cfg["alpha"])
    use_log_scores = bool(cfg["use_log_scores"])
    score_scale = float(cfg["score_scale"])
    use_rank_scores = bool(cfg.get("use_rank_scores", False))
    objective_scale = int(cfg.get("objective_scale", DEFAULT_CONFIG["objective_scale"]))

    bottle_status = load_bottle_status_map(metadata_csv)
    include_bottles = not bool(cfg.get("use_ingredients", False))
    cocktails_raw = load_cocktails(
        index_csv, bottle_status, include_bottles=include_bottles
    )

    scores = load_scores(scores_csv)
    cocktails = compute_cocktails(
        cocktails_raw,
        scores,
        default_score=1.0,
        use_rank_scores=use_rank_scores,
        use_log_scores=use_log_scores,
        score_scale=score_scale,
    )

    ingredient_to_cocktails: dict[str, list[str]] = defaultdict(list)
    for c in cocktails:
        for ing in c.ingredients:
            ingredient_to_cocktails[ing].append(c.name)

    ingredient_set = sorted(ingredient_to_cocktails.keys())

    cooccurrence_counts: dict[tuple[str, str], int] = defaultdict(int)
    for c in cocktails:
        unique_ings = sorted(set(c.ingredients))
        for a, b in combinations(unique_ings, 2):
            cooccurrence_counts[(a, b)] += 1

    (
        selected_ingredients,
        enabled_cocktails,
        enabled_weight,
        ingredient_marginals,
    ) = solve(
        cocktails=cocktails,
        max_bottles=max_bottles,
        alpha=alpha,
        objective_scale=objective_scale,
    )

    ingredient_frequency = {
        ing: len(ingredient_to_cocktails.get(ing, [])) for ing in ingredient_set
    }
    coverage_progression = compute_coverage_progression(
        cocktails=cocktails,
        selected_ingredients=selected_ingredients,
        ingredient_frequency=ingredient_frequency,
    )

    write_intermediate_artifacts(
        output_dir=output_dir,
        cocktails=cocktails,
        ingredient_set=ingredient_set,
        ingredient_to_cocktails=ingredient_to_cocktails,
        cooccurrence_counts=dict(cooccurrence_counts),
        selected_ingredients=selected_ingredients,
        enabled_cocktails=enabled_cocktails,
        ingredient_marginals=ingredient_marginals,
        coverage_progression=coverage_progression,
    )

    print("Selected ingredients (bottles):")
    for ing in selected_ingredients:
        print(f"- {ing}")

    print("\nAchievable cocktails:")
    for name in enabled_cocktails:
        print(f"- {name}")

    print(f"\nTotal cocktails: {len(enabled_cocktails)}")
    print(f"Weighted score: {enabled_weight:.6f}")


if __name__ == "__main__":
    main()
