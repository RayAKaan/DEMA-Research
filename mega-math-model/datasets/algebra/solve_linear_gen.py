import random
import json
from sympy import Eq, solve, symbols

x = symbols('x')

def generate_example():
    a = random.randint(1,10)
    b = random.randint(-10,10)
    c = random.randint(1,10)
    d = random.randint(-10,10)

    expr = f"{a}*x + {b} = {c}*x + {d}"

    # Split into LHS and RHS
    lhs, rhs = expr.split("=")
    
    try:
        # Build a SymPy equation
        equation = Eq(eval(lhs), eval(rhs))
        sol = solve(equation, x)

        if len(sol) == 0:
            out = "NoSolution"
        else:
            out = str(sol[0])
    except Exception:
        out = "NoSolution"

    return {"input": expr, "output": out}

def generate_dataset(n=20000, out_path=None):
    data = [generate_example() for _ in range(n)]
    if out_path:
        with open(out_path, "w") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")
    return data

if __name__ == '__main__':
    generate_dataset(20000, "solve_linear.jsonl")
