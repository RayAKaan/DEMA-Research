import random
import json
from sympy import expand, symbols
from sympy.core.sympify import SympifyError

x = symbols('x')

def generate_example():
    # (ax + b)(cx + d)
    a,b,c,d = [random.randint(1,10) for _ in range(4)]
    expr = f"({a}*x+{b})*({c}*x+{d})"
    try:
        out = str(expand(expr))
    except SympifyError:
        out = expr
    return {"input": expr, "output": out}

def generate_dataset(n=30000, out_path=None):
    data = [generate_example() for _ in range(n)]
    if out_path:
        with open(out_path, 'w') as f:
            for d in data:
                f.write(json.dumps(d) + "\n")
    return data

if __name__ == '__main__':
    generate_dataset(30000, "expand_polynomial.jsonl")
