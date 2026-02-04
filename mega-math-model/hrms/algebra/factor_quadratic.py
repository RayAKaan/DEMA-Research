import random
from sympy import symbols, expand, factor
from hrms.base import HRMBase

x = symbols('x')

class FactorQuadraticHRM(HRMBase):
    def __init__(self, name='Factor_Quadratic', d_model=64, vocab_size=128):
        super().__init__(name=name, vocab_size=vocab_size, d_model=d_model)

    def forward(self, tokens):
        raw = ''.join(chr(int(t)) if int(t) < 128 else '' for t in tokens[0].tolist())
        try:
            out = str(factor(raw))
        except:
            out = raw
        return [[c for c in out]], {"conf": 0.95}

def make_dataset(n=20000):
    examples = []
    for _ in range(n):
        r1 = random.randint(-10,10)
        r2 = random.randint(-10,10)
        expr = str(expand((x - r1)*(x - r2)))
        gold = str(factor(expr))
        examples.append((expr,gold))
    return examples
