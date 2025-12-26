import random
from hrms.base import HRMBase

class Divide1DigitHRM(HRMBase):
    def __init__(self, name="Divide_1digit", d_model=64, vocab_size=128):
        super().__init__(name=name, vocab_size=vocab_size, d_model=d_model)

    def forward(self, tokens):
        try:
            raw = "".join(chr(int(t)) for t in tokens[0] if int(t) < 128)
        except:
            raw = str(tokens)

        raw = raw.replace("÷", "/")

        try:
            a_s, b_s = raw.split("/")
            a = int(a_s)
            b = int(b_s)
            if b == 0:
                out = "Undefined"
            else:
                out = f"{a//b} rem {a%b}"
        except:
            out = raw

        return [[c for c in out]], {"conf": 0.90}

def make_dataset(n=20000):
    ds = []
    for _ in range(n):
        a = random.randint(0,9)
        b = random.randint(1,9)
        ds.append((f"{a}/{b}", f"{a//b} rem {a%b}"))
    return ds
