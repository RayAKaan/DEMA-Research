from typing import List

VOCAB = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-*/^()=,. x")
PAD = "<pad>"
UNK = "<unk>"

class SimpleTokenizer:
    def __init__(self):
        self.vocab = [PAD, UNK] + VOCAB
        self.i2s = {i: s for i, s in enumerate(self.vocab)}
        self.s2i = {s: i for i, s in self.i2s.items()}

    def tokenize(self, s: str):
        return [self.s2i.get(ch, self.s2i[UNK]) for ch in s.strip()]

    def detokenize(self, tokens):
        return "".join(self.i2s.get(int(t), UNK) for t in tokens).strip()

tokenizer = SimpleTokenizer()

def tokenize(s): return tokenizer.tokenize(s)
def detokenize(t): return tokenizer.detokenize(t)
