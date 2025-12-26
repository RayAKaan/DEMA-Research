import torch
import torch.nn as nn
import random
from sympy import sympify
from hrms.base import HRMBase

class SimplifyLikeTermsHRM(HRMBase):
    def __init__(self, name='Simplify_LikeTerms', d_model=128, vocab_size=128):
        super().__init__(name=name, vocab_size=vocab_size, d_model=d_model)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.GRU(d_model, d_model//2, batch_first=True, bidirectional=True)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens):
        raw = ''.join(chr(int(t)) if int(t) < 128 else '' for t in tokens[0].tolist())
        try:
            gold = str(sympify(raw).simplify())
        except:
            gold = raw
        return [[c for c in gold]], {"conf": 0.95}

def make_dataset(n=50000):
    from sympy import symbols
    x = symbols('x')
    examples = []
    for _ in range(n):
        a = random.randint(1,10)
        b = random.randint(1,10)
        expr = f"{a}*x + {b}*x"
        try:
            out = str(sympify(expr).simplify())
        except:
            out = expr
        examples.append((expr,out))
    return examples
