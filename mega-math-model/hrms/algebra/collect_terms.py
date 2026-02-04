import random
from sympy import sympify, collect, symbols
from hrms.base import HRMBase

x = symbols('x')

class CollectTermsHRM(HRMBase):
    def __init__(self, name='Collect_Terms', d_model=64, vocab_size=128):
        super().__init__(name=name, vocab_size=vocab_size, d_model=d_model)

    def forward(self, tokens):
        raw = ''.join(chr(int(t)) if int(t) < 128 else '' for t in tokens[0].tolist())
        try:
            e = sympify(raw)
            out = str(collect(e, x))
        except:
            out = raw
        return [[c for c in out]], {"conf": 0.9}

def make_dataset(n=20000):
    examples = []
    for _ in range(n):
        a,b,c = [random.randint(1,10) for _ in range(3)]
        expr = f"{a}*x + {b} + {c}*x"
        try:
            gold = str(sympify(expr))
        except:
            gold = expr
        examples.append((expr,gold))
    return examples
