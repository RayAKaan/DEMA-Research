import random
import json
from sympy import sympify, simplify
from sympy.core.sympify import SympifyError

def generate_example():
    # random simple like terms: ax + bx
    a = random.randint(1,10)
    b = random.randint(1,10)
    expr = f"{a}*x + {b}*x"
    try:
        out = str(simplify(expr))
    except SympifyError:
        out = expr
    return {"input": expr, "output": out}

def generate_dataset(n=50000, out_path=None):
    data = [generate_example() for _ in range(n)]
    if out_path:
        with open(out_path, "w") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")
    return data

if __name__ == '__main__':
    generate_dataset(50000, "simplify_like_terms.jsonl")
