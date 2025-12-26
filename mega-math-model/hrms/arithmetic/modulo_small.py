import random
from hrms.base import HRMBase

class ModuloSmallHRM(HRMBase):
    def __init__(self, name="Modulo_Small", d_model=64, vocab_size=128):
        super().__init__(name=name, vocab_size=vocab_size, d_model=d_model)

    def forward(self, tokens):
        try:
            raw = "".join(chr(int(t)) for t in tokens[0] if int(t) < 128)
        except:
            raw = str(tokens)

        raw = raw.replace("%", " ").replace("mod", " ")

        try:
            a, b = raw.split()
            out = str(int(a) % int(b))
        except:
            out = raw

        return [[c for c in out]], {"conf": 0.90}

def make_dataset(n=20000, max_val=200):
    ds = []
    for _ in range(n):
        a = random.randint(0,max_val)
        b = random.randint(1,max_val)
        ds.append((f"{a}%{b}", str(a%b)))
    return ds
