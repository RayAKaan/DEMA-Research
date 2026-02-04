import random
from sympy import symbols, diff
from hrms.base import HRMBase

x = symbols('x')

class DerivativePowerHRM(HRMBase):
    def __init__(self, name='Derivative_Power', d_model=64, vocab_size=128):
        super().__init__(name=name, vocab_size=vocab_size, d_model=d_model)

    def forward(self, tokens):
        raw = ''.join(chr(int(t)) if int(t) < 128 else '' for t in tokens[0].tolist())
        try:
            out = str(diff(raw, x))
        except:
            out = raw
        return [[c for c in out]], {"conf": 0.95}

def make_dataset(n=30000):
    examples = []
    for _ in range(n):
        coeff = random.randint(1,10)
        p = random.randint(1,6)
        expr = f"{coeff}*x**{p}"
        gold = str(diff(expr, x))
        examples.append((expr,gold))
    return examples
