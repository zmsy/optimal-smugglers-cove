"""Create a minimal SVG line plot from ingredient_max_vs_drinks.csv.

X axis: max_ingredients
Y axis: num_possible_drinks

This is intentionally dependency-free to keep setup simple.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


CSV_PATH = Path(__file__).resolve().parent.parent / "ingredient_max_vs_drinks.csv"
OUT_SVG_PATH = (
    Path(__file__).resolve().parent.parent / "output" / "ingredient_max_vs_drinks.svg"
)


@dataclass(frozen=True)
class PlotBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float


def _nice_step(step: float) -> float:
    """Round step to a 1/2/5 * 10^k style interval."""
    if step <= 0:
        return 1
    power = 10 ** math.floor(math.log10(step))
    scaled = step / power
    if scaled <= 1:
        return 1 * power
    if scaled <= 2:
        return 2 * power
    if scaled <= 5:
        return 5 * power
    return 10 * power


def _ticks(lo: float, hi: float, target_count: int) -> List[float]:
    span = hi - lo
    if span <= 0:
        return [lo]
    raw_step = span / max(1, target_count)
    step = _nice_step(raw_step)

    start = math.floor(lo / step) * step
    end = math.ceil(hi / step) * step

    ticks: List[float] = []
    t = start
    # Guard against infinite loops due to float quirks.
    for _ in range(10_000):
        if t > end + 1e-9:
            break
        if t >= lo - 1e-9 and t <= hi + 1e-9:
            ticks.append(t)
        t += step

    # If we ended up with too few (e.g. lo/hi already snapped), widen.
    if len(ticks) < 2:
        ticks = [lo, hi]

    return ticks


def _read_points(csv_path: Path) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = float(row["max_ingredients"])
            y = float(row["num_possible_drinks"])
            points.append((x, y))

    points.sort(key=lambda p: p[0])
    return points


def _compute_bounds(points: List[Tuple[float, float]]) -> PlotBounds:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    # Add a small padding so the line doesn't sit on the border.
    x_pad = (x_max - x_min) * 0.03 if x_max > x_min else 1
    y_pad = (y_max - y_min) * 0.08 if y_max > y_min else 1

    return PlotBounds(
        x_min=x_min - x_pad,
        x_max=x_max + x_pad,
        y_min=y_min - y_pad,
        y_max=y_max + y_pad,
    )


def _fmt_tick(v: float) -> str:
    # Keep integers as integers; otherwise use a compact format.
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.2f}".rstrip("0").rstrip(".")


def to_svg(
    points: List[Tuple[float, float]], *, width: int = 900, height: int = 500
) -> str:
    bounds = _compute_bounds(points)

    margin_left = 85
    margin_right = 20
    margin_top = 25
    margin_bottom = 55

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    def x_to_px(x: float) -> float:
        return margin_left + (x - bounds.x_min) / (bounds.x_max - bounds.x_min) * plot_w

    def y_to_px(y: float) -> float:
        # SVG y grows downward; invert.
        return margin_top + (bounds.y_max - y) / (bounds.y_max - bounds.y_min) * plot_h

    xticks = _ticks(bounds.x_min, bounds.x_max, target_count=8)
    yticks = _ticks(bounds.y_min, bounds.y_max, target_count=6)

    # Build polyline points.
    poly_points = " ".join(f"{x_to_px(x):.2f},{y_to_px(y):.2f}" for x, y in points)

    axis_color = "currentColor"
    label_color = "currentColor"
    line_color = "currentColor"

    # Let the blog's CSS choose the font (works when the SVG is inline).
    font_family = "inherit"

    # Font sizes for axis text.
    # User request: make axis text 1.5x bigger.
    tick_font_size = 12 * 1.5
    axis_label_font_size = 13 * 1.5

    # Minimal: no grid lines (ticks only). We still render tick marks for readability.
    parts: List[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="max_ingredients vs num_possible_drinks line plot" style="color: inherit; font-family: inherit;">'
    )
    # Transparent background (no rect fill).

    # Axes
    x_axis_y = y_to_px(0 if bounds.y_min <= 0 <= bounds.y_max else bounds.y_min)
    y_axis_x = x_to_px(bounds.x_min)

    # Clamp axes to plot area.
    x_axis_y = max(margin_top, min(margin_top + plot_h, x_axis_y))
    y_axis_x = max(margin_left, min(margin_left + plot_w, y_axis_x))

    parts.append(
        f'<line x1="{y_axis_x:.2f}" y1="{margin_top}" x2="{y_axis_x:.2f}" y2="{margin_top + plot_h:.2f}" stroke="{axis_color}" stroke-width="1" />'
    )
    parts.append(
        f'<line x1="{margin_left}" y1="{x_axis_y:.2f}" x2="{margin_left + plot_w:.2f}" y2="{x_axis_y:.2f}" stroke="{axis_color}" stroke-width="1" />'
    )

    # X ticks + labels
    for t in xticks:
        px = x_to_px(t)
        parts.append(
            f'<line x1="{px:.2f}" y1="{x_axis_y:.2f}" x2="{px:.2f}" y2="{x_axis_y + 6:.2f}" stroke="{axis_color}" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{px:.2f}" y="{x_axis_y + 24:.2f}" text-anchor="middle" fill="{label_color}" font-family="{font_family}" font-size="{tick_font_size}">{_fmt_tick(t)}</text>'
        )

    # Y ticks + labels
    for t in yticks:
        py = y_to_px(t)
        parts.append(
            f'<line x1="{y_axis_x:.2f}" y1="{py:.2f}" x2="{y_axis_x - 6:.2f}" y2="{py:.2f}" stroke="{axis_color}" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{y_axis_x - 10:.2f}" y="{py + 4:.2f}" text-anchor="end" fill="{label_color}" font-family="{font_family}" font-size="{tick_font_size}">{_fmt_tick(t)}</text>'
        )

    # Axis labels
    parts.append(
        f'<text x="{margin_left + plot_w / 2:.2f}" y="{height - 18}" text-anchor="middle" fill="{label_color}" font-family="{font_family}" font-size="{axis_label_font_size}">Ingredients</text>'
    )
    parts.append(
        f'<text x="18" y="{margin_top + plot_h / 2:.2f}" text-anchor="start" fill="{label_color}" font-family="{font_family}" font-size="{axis_label_font_size}" transform="rotate(-90 18 {margin_top + plot_h / 2:.2f})">Drinks</text>'
    )

    # Plot line
    parts.append(
        f'<polyline points="{poly_points}" fill="none" stroke="{line_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />'
    )

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    points = _read_points(CSV_PATH)
    OUT_SVG_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_SVG_PATH.write_text(to_svg(points), encoding="utf-8")


if __name__ == "__main__":
    main()
