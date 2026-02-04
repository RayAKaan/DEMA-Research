import random
from hrms.base import HRMBase

class SubtractNDigitHRM(HRMBase):
    def __init__(self, name="Subtract_Ndigit", d_model=128, vocab_size=128):
        super().__init__(name=name, vocab_size=vocab_size, d_model=d_model)

    def forward(self, tokens):
        try:
            raw = "".join(chr(int(t)) for t in tokens[0] if int(t) < 128)
        except:
            raw = str(tokens)

        try:
            a_s, b_s = raw.split("-")
            out = str(int(a_s) - int(b_s))
        except:
            out = raw

        return [[c for c in out]], {"conf": 0.90}

def make_dataset(n=50000, digits=3):
    ds = []
    lo = 10**(digits-1)
    hi = 10**digits - 1
    for _ in range(n):
        a = random.randint(lo,hi)
        b = random.randint(lo,hi)
        ds.append((f"{a}-{b}", str(a-b)))
    return ds
