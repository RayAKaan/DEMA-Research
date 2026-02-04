import random
import json
from sympy import symbols, diff

x = symbols('x')

def generate_example():
    a, b, c, d = [random.randint(1,6) for _ in range(4)]
    expr = f"({a}*x+{b})*({c}*x+{d})"
    out = str(diff(expr, x))
    return {"input": expr, "output": out}

def generate_dataset(n=20000, out_path=None):
    data = [generate_example() for _ in range(n)]
    if out_path:
        with open(out_path,'w') as f:
            for d in data: f.write(json.dumps(d) + "\n")
    return data

if __name__ == "__main__":
    generate_dataset(20000, "derivative_product.jsonl")
