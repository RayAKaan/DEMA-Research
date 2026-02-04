import random
import json
from sympy import sympify, collect, symbols
from sympy.core.sympify import SympifyError

x = symbols('x')

def generate_example():
    # random expression with like terms out of order
    a,b,c = [random.randint(1,10) for _ in range(3)]
    expr = f"{a}*x + {b} + {c}*x"
    try:
        out = str(collect(sympify(expr), x))
    except SympifyError:
        out = expr
    return {"input": expr, "output": out}

def generate_dataset(n=20000, out_path=None):
    data = [generate_example() for _ in range(n)]
    if out_path:
        with open(out_path,'w') as f:
            for d in data:
                f.write(json.dumps(d) + "\n")
    return data

if __name__ == '__main__':
    generate_dataset(20000, "collect_terms.jsonl")
