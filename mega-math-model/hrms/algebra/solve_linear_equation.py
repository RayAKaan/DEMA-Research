import random
from sympy import symbols, solve
from hrms.base import HRMBase

x = symbols('x')

class SolveLinearEquationHRM(HRMBase):
    def __init__(self, name='Solve_Linear_Equation', d_model=64, vocab_size=128):
        super().__init__(name=name, vocab_size=vocab_size, d_model=d_model)

    def forward(self, tokens):
        raw = ''.join(chr(int(t)) if int(t) < 128 else '' for t in tokens[0].tolist())
        try:
            sol = solve(raw, x)
            out = str(sol[0]) if sol else 'NoSolution'
        except:
            out = raw
        return [[c for c in out]], {"conf": 0.9}

def make_dataset(n=20000):
    examples = []
    for _ in range(n):
        a = random.randint(1,10)
        b = random.randint(-10,10)
        c = random.randint(1,10)
        d = random.randint(-10,10)
        expr = f"{a}*x + {b} = {c}*x + {d}"
        try:
            sol = solve(expr, x)
            gold = str(sol[0]) if sol else 'NoSolution'
        except:
            gold = 'NoSolution'
        examples.append((expr,gold))
    return examples
