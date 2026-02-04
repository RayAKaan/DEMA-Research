import torch
import torch.nn as nn

class HRMBase(nn.Module):
    def __init__(self, name: str, vocab_size: int = 128, d_model: int = 128):
        super().__init__()
        self.name = name
        self.vocab_size = vocab_size
        self.d_model = d_model

    def forward(self, tokens):
        raise NotImplementedError("HRM must implement forward()")

    def inference(self, raw_input, tokenizer, device='cpu'):
        tokens = tokenizer.tokenize(raw_input)
        tokens_t = torch.tensor([tokens], dtype=torch.long, device=device)
        self.eval()
        with torch.no_grad():
            out_tokens, meta = self.forward(tokens_t)
        out_str = tokenizer.detokenize(out_tokens[0])
        return out_str, meta.get('conf', 1.0), meta
