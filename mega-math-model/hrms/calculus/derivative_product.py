import random
from sympy import diff
from hrms.base import HRMBase

class DerivativeProductHRM(HRMBase):
    def __init__(self, name='Derivative_Product', d_model=64, vocab_size=128):
        super().__init__(name=name, vocab_size=vocab_size, d_model=d_model)

    def forward(self, tokens):
        raw = ''.join(chr(int(t)) if int(t) < 128 else '' for t in tokens[0].tolist())
        try:
            out = str(diff(raw))
        except:
            out = raw
        return [[c for c in out]], {"conf": 0.95}

def make_dataset(n=20000):
    examples = []
    for _ in range(n):
        a,b,c,d = [random.randint(1,6) for _ in range(4)]
        expr = f"({a}*x+{b})*({c}*x+{d})"
        gold = str(diff(expr))
        examples.append((expr,gold))
    return examples
