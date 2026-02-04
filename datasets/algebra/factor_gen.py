import random
import json
from sympy import symbols, expand, factor

x = symbols('x')

def generate_example():
    # generate 2 integer roots
    r1 = random.randint(-10,10)
    r2 = random.randint(-10,10)
    expr = str(expand((x - r1)*(x - r2)))
    out = str(factor(expr))
    return {"input": expr, "output": out}

def generate_dataset(n=20000, out_path=None):
    data = [generate_example() for _ in range(n)]
    if out_path:
        with open(out_path,'w') as f:
            for d in data:
                f.write(json.dumps(d) + "\n")
    return data

if __name__ == '__main__':
    generate_dataset(20000, "factor_quadratic.jsonl")
