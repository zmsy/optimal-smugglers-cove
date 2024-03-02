"""
Invoke CSP solver for the given constraint. Some for this dataset are:

- Most drinks for the amount of bottles
- Best bang for your buck
"""


from ortools.sat.python import cp_model
from csv_data import get_drinks


def solve_optimal_drinks(num: int) -> None:
    """
    Solve for the optimal set of drinks for each number of bottles.
    """
    model = cp_model.CpModel()
    drinks = get_drinks()

    """
    Add your drinks and your ingredients as variables, and defined the relation
    between them such that a drink variable is only true if all of its assigned
    ingredients are true.
    """
    ingredients = set(
        ingredient for drink in drinks for ingredient in drink.ingredients
    )
    ingredient_vars = {x: model.NewBoolVar(x) for x in ingredients}

    drink_vars = []
    for drink in drinks:
        required_ingredient_vars = [ingredient_vars[x] for x in drink.ingredients]
        drink_var = model.NewBoolVar(drink.name)
        model.AddBoolAnd(required_ingredient_vars).OnlyEnforceIf(drink_var)  # type: ignore
        drink_vars.append(drink_var)  # type: ignore

    """
    Define the model's objective and constraints.
    """
    # restrict the solution to *just* the number of ingredients passed in
    model.Add(sum(ingredient_vars.values()) == num)

    # Maximize the number of drinks that can be made
    model.Maximize(sum(drink_vars))  # type: ignore

    """
    Solve and output!
    """

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    solutions = []
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(
            f"Max drinks with {num} ingredients ({status}): {int(solver.ObjectiveValue())}"
        )
        solution = {
            "num": num,
            "drinks_possible": int(solver.ObjectiveValue()),
            "drinks": [],
            "ingredients": [],
        }
        print("Ingredients to buy:")
        for ingredient, var in ingredient_vars.items():
            if solver.Value(var):
                print(f"- {ingredient}")
                solution["ingredients"].append(ingredient)
        print("\nDrinks you can make:")
        for drink in drinks:
            if solver.Value(drink_vars[drinks.index(drink)]):
                print(f"- {drink.name}")
                solution["drinks"].append(drink.name)

        solutions.append(solution)

    else:
        print("No solution found.")


for i in range(3, 72):
    solve_optimal_drinks(i)
