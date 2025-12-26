import random
from hrms.base import HRMBase

class Subtract1DigitHRM(HRMBase):
    def __init__(self, name="Subtract_1digit", d_model=64, vocab_size=128):
        super().__init__(name=name, vocab_size=vocab_size, d_model=d_model)

    def forward(self, tokens):
        # decode input tokens to string
        try:
            raw = "".join(chr(int(t)) for t in tokens[0] if int(t) < 128)
        except:
            raw = str(tokens)

        try:
            a_s, b_s = raw.split("-")
            a = int(a_s.strip())
            b = int(b_s.strip())
            out = str(a - b)
        except:
            out = raw

        return [[c for c in out]], {"conf": 0.95}

def make_dataset(n=20000):
    ds = []
    for _ in range(n):
        a = random.randint(0,9)
        b = random.randint(0,9)
        ds.append((f"{a}-{b}", str(a-b)))
    return ds
