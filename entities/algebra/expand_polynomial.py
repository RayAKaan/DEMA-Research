import random
from sympy import symbols, expand
from hrms.base import HRMBase

x = symbols('x')

class ExpandPolynomialHRM(HRMBase):
    def __init__(self, name='Expand_Polynomial', d_model=128, vocab_size=128):
        super().__init__(name=name, vocab_size=vocab_size, d_model=d_model)

    def forward(self, tokens):
        raw = ''.join(chr(int(t)) if int(t) < 128 else '' for t in tokens[0].tolist())
        try:
            out = str(expand(raw))
        except:
            out = raw
        return [[c for c in out]], {"conf": 0.95}

def make_dataset(n=30000):
    examples = []
    for _ in range(n):
        a,b,c,d = [random.randint(1,10) for _ in range(4)]
        expr = f"({a}*x+{b})*({c}*x+{d})"
        try:
            out = str(expand(expr))
        except:
            out = expr
        examples.append((expr,out))
    return examples
