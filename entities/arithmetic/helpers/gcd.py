import torch
import torch.nn as nn
import random
import math
from hrms.base import HRMBase

class GCDHRM(HRMBase):
    def __init__(self, name='GCD_Euclid', d_model=64, vocab_size=128):
        super().__init__(name=name, vocab_size=vocab_size, d_model=d_model)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model,128), nn.ReLU(),
            nn.Linear(128,64), nn.ReLU(),
            nn.Linear(64,8)
        )

    def forward(self, tokens):
        x = self.embed(tokens).mean(dim=1)
        logits = self.net(x)
        preds = logits.argmax(dim=-1).tolist()
        return [[str(p)] for p in preds], {"conf": 0.7}

def make_dataset(n=20000, max_v=2000):
    examples = []
    for _ in range(n):
        a = random.randint(1, max_v)
        b = random.randint(1, max_v)
        examples.append((f"gcd({a},{b})", str(math.gcd(a,b))))
    return examples
