import random
import json
from sympy import symbols, diff

x = symbols('x')

def generate_example():
    coeff = random.randint(1,10)
    p = random.randint(1,6)
    expr = f"{coeff}*x**{p}"
    out = str(diff(expr, x))
    return {"input": expr, "output": out}

def generate_dataset(n=30000, out_path=None):
    data = [generate_example() for _ in range(n)]
    if out_path:
        with open(out_path, 'w') as f:
            for d in data: f.write(json.dumps(d) + "\n")
    return data

if __name__ == "__main__":
    generate_dataset(30000, "derivative_power.jsonl")
