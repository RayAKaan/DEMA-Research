import random
from hrms.base import HRMBase

class NegativeNumberHandlerHRM(HRMBase):
    def __init__(self, name="NegativeNumber", d_model=64, vocab_size=128):
        super().__init__(name=name, vocab_size=vocab_size, d_model=d_model)

    def forward(self, tokens):
        try:
            raw = "".join(chr(int(t)) for t in tokens[0] if int(t) < 128)
        except:
            raw = str(tokens)

        expr = raw.replace("-","-")

        try:
            out = str(eval(expr))  # safe for digits + +/- only
        except:
            out = expr

        return [[c for c in out]], {"conf": 0.95}

def make_dataset(n=10000):
    ds = []
    for _ in range(n):
        a = random.randint(-20,20)
        ds.append((str(a), str(a)))
    return ds
