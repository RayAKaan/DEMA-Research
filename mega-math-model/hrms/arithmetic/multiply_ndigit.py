import torch
import torch.nn as nn
import random
from hrms.base import HRMBase

class MulNDigitHRM(HRMBase):
    def __init__(self, name='Multiply_Ndigit', d_model=128, vocab_size=128):
        super().__init__(name=name, vocab_size=vocab_size, d_model=d_model)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.LSTM(d_model, d_model//2, batch_first=True, bidirectional=True)
        self.head = nn.Linear(d_model, 32)

    def forward(self, tokens):
        x = self.embed(tokens)
        out, _ = self.rnn(x)
        pooled = out.mean(dim=1)
        logits = self.head(pooled)
        pred = logits.argmax(dim=-1).tolist()
        out_tokens = [[str(p % 10)] for p in pred]
        return out_tokens, {"conf": 0.8}

def make_dataset(n=30000, digits=3):
    examples = []
    for _ in range(n):
        a = random.randint(10**(digits-1), 10**digits - 1)
        b = random.randint(10**(digits-1), 10**digits - 1)
        examples.append((f"{a}*{b}", str(a*b)))
    return examples
