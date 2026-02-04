import torch
import torch.nn as nn
import random
from hrms.base import HRMBase

class Add1DigitHRM(HRMBase):
    def __init__(self, name='Add_1digit', d_model=64, vocab_size=128):
        super().__init__(name=name, vocab_size=vocab_size, d_model=d_model)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 10)
        )

    def forward(self, tokens):
        x = self.embed(tokens).mean(dim=1)
        logits = self.net(x)
        preds = logits.argmax(dim=-1).tolist()
        out_tokens = [[str(p)] for p in preds]
        return out_tokens, {"conf": 0.9}

def make_dataset(n=20000):
    examples = []
    for _ in range(n):
        a = random.randint(0, 9)
        b = random.randint(0, 9)
        examples.append((f"{a}+{b}", str(a+b)))
    return examples
