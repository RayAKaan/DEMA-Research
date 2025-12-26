#!/usr/bin/env python3
"""
training/train_hrm.py

Generic training script for HRMs.

Features:
- dynamic import of HRM class: --hrm module_path:ClassName
- uses HRM.make_dataset(n) if available to synthesize data
- uses project tokenizer (hrms.utils.tokenizer)
- if HRM has learnable params, can train it directly
- fallback: trains a tiny Seq2Seq model (SimpleSeq2Seq) on the same data
- saves checkpoints to --save-dir
"""

import os
import argparse
import importlib
import json
from typing import List, Tuple
import random
from pathlib import Path
import math
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Use project tokenizer
from hrms.utils.tokenizer import tokenizer, tokenize, detokenize

# -------------------------
# Utilities
# -------------------------
def dynamic_import(path: str):
    """
    path: "hrms.arithmetic.add_ndigit:AddNDigitHRM"
    returns: class
    """
    if ":" not in path:
        raise ValueError("Invalid hrm path. Use module:ClassName format.")
    mod_path, cls_name = path.split(":")
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)
    return cls

# -------------------------
# Dataset wrapper
# -------------------------
class HRMDataset(Dataset):
    def __init__(self, examples: List[Tuple[str,str]], tok, max_input_len=64, max_output_len=64):
        self.examples = examples
        self.tok = tok
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.pad_id = 0  # tokenizer uses PAD at index 0

    def __len__(self):
        return len(self.examples)

    def _encode(self, s, max_len):
        ids = self.tok.tokenize(s)
        if len(ids) >= max_len:
            ids = ids[: max_len - 1]
        # pad
        ids = ids + [self.pad_id] * (max_len - len(ids))
        return ids

    def __getitem__(self, idx):
        inp, out = self.examples[idx]
        src_ids = self._encode(inp, self.max_input_len)
        tgt_ids = self._encode(out, self.max_output_len)
        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "raw": (inp, out)
        }

def collate_fn(batch):
    src = torch.stack([b["src_ids"] for b in batch], dim=0)
    tgt = torch.stack([b["tgt_ids"] for b in batch], dim=0)
    raws = [b["raw"] for b in batch]
    return {"src": src, "tgt": tgt, "raw": raws}

# -------------------------
# Simple Seq2Seq fallback model
# -------------------------
class SimpleSeq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, enc_hid=256, dec_hid=256, pad_id=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.encoder = nn.GRU(embed_dim, enc_hid // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.decoder = nn.GRU(embed_dim, dec_hid, num_layers=1, batch_first=True)
        self.out = nn.Linear(dec_hid, vocab_size)
        self.vocab_size = vocab_size
        self.pad_id = pad_id

    def encode(self, src):
        # src: [B, T]
        emb = self.embed(src)  # [B, T, E]
        enc_out, h = self.encoder(emb)  # enc_out [B,T,enc_hid]
        # combine bidir hidden states to init decoder
        if isinstance(h, torch.Tensor):
            # h shape: (num_layers * num_directions, B, hid//2)
            # for num_layers=1 -> shape (2, B, hid//2)
            h_cat = torch.cat([h[0], h[1]], dim=-1)  # [B, enc_hid]
            # expand to (1, B, dec_hid)
            h0 = h_cat.unsqueeze(0)
        else:
            h0 = None
        return enc_out, h0

    def forward(self, src, tgt=None, teacher_forcing_ratio=0.5, max_len=64):
        # training: use teacher forcing with tgt provided
        B = src.size(0)
        enc_out, h0 = self.encode(src)  # h0 may be None; decoder will init zeros if so
        device = src.device
        if tgt is not None:
            # use teacher forcing
            emb_tgt = self.embed(tgt)  # [B, T, E]
            dec_out, _ = self.decoder(emb_tgt, h0)  # [B,T,dec_hid]
            logits = self.out(dec_out)  # [B,T,V]
            return logits
        else:
            # inference greedy decode
            inputs = torch.full((B,1), self.pad_id, dtype=torch.long, device=device)  # start tokens unknown -> pad
            outputs = []
            hidden = h0
            for _ in range(max_len):
                emb = self.embed(inputs)[:, -1:, :]  # [B,1,E]
                dec_out, hidden = self.decoder(emb, hidden)
                logits = self.out(dec_out)  # [B,1,V]
                next_token = logits.argmax(dim=-1)  # [B,1]
                outputs.append(next_token)
                inputs = torch.cat([inputs, next_token], dim=1)
            out_seq = torch.cat(outputs, dim=1)  # [B, T]
            return out_seq

# -------------------------
# Training loop
# -------------------------
def train_loop(model, dataloader, optimizer, device, scaler=None, loss_fn=None):
    model.train()
    total_loss = 0.0
    n_tokens = 0
    for batch in dataloader:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(src, tgt)
                # logits [B, T, V]; tgt [B, T]
                loss = loss_fn(logits.view(-1, logits.size(-1)), tgt.view(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(src, tgt)
            loss = loss_fn(logits.view(-1, logits.size(-1)), tgt.view(-1))
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * src.size(0)
        n_tokens += src.size(0)
    return total_loss / max(1, n_tokens)

def eval_loop(model, dataloader, device, loss_fn=None):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in dataloader:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            logits = model(src, tgt)
            loss = loss_fn(logits.view(-1, logits.size(-1)), tgt.view(-1))
            total_loss += loss.item() * src.size(0)
            n += src.size(0)
    return total_loss / max(1, n)

# -------------------------
# Entry point
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hrm", required=True, help="HRM module path, e.g. hrms.arithmetic.add_ndigit:AddNDigitHRM")
    parser.add_argument("--save-dir", default="registry/hrm_versions/tmp_hrm", help="checkpoint save dir")
    parser.add_argument("--dataset-size", type=int, default=20000, help="number of synthetic examples to generate")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-input-len", type=int, default=64)
    parser.add_argument("--max-output-len", type=int, default=64)
    parser.add_argument("--use-wrapper", action="store_true", help="force training with internal Seq2Seq wrapper")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    # dynamic import HRM class
    HRMClass = dynamic_import(args.hrm)
    # instantiate (may be a symbolic wrapper HRM)
    try:
        hrm_obj = HRMClass()
    except TypeError:
        hrm_obj = HRMClass(name=getattr(HRMClass, "__name__", "hrm"))
    train_dataset = None

    # try to get dataset from HRM.make_dataset
    if hasattr(HRMClass, "make_dataset"):
        print("[data] Using HRMClass.make_dataset")
        examples = HRMClass.make_dataset(args.dataset_size)
    elif hasattr(hrm_obj, "make_dataset"):
        print("[data] Using hrm_obj.make_dataset()")
        examples = hrm_obj.make_dataset(args.dataset_size)
    else:
        # try to import generator module with convention datasets/...
        # fallback: try to call a function named generate_<hrmname>
        mod_name = args.hrm.split(":")[0]
        try:
            mod = importlib.import_module(mod_name)
            gen_name = f"make_dataset"
            if hasattr(mod, gen_name):
                examples = getattr(mod, gen_name)(args.dataset_size)
            else:
                raise RuntimeError("No dataset generator found.")
        except Exception as e:
            raise RuntimeError("Cannot find dataset generator for HRM. Provide HRM with make_dataset method.") from e

    # split
    random.shuffle(examples)
    n = len(examples)
    ntrain = int(n * 0.8)
    nval = int(n * 0.1)
    train_examples = examples[:ntrain]
    val_examples = examples[ntrain:ntrain+nval]
    test_examples = examples[ntrain+nval:]

    print(f"[data] total {n} examples -> train={len(train_examples)} val={len(val_examples)} test={len(test_examples)}")

    # build datasets
    ds_train = HRMDataset(train_examples, tokenizer, max_input_len=args.max_input_len, max_output_len=args.max_output_len)
    ds_val = HRMDataset(val_examples, tokenizer, max_input_len=args.max_input_len, max_output_len=args.max_output_len)
    train_dl = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device(args.device)
    use_fp16 = (device.type == "cuda")

    # Decide model: use HRM if trainable and not forcing wrapper
    trainable_hrm = (len(list(hrm_obj.parameters())) > 0) if hasattr(hrm_obj, "parameters") else False
    if trainable_hrm and not args.use_wrapper:
        print("[model] Training HRM model directly.")
        model = hrm_obj.to(device)
        # If HRM forward returns logits when called with tokens, we expect shape [B,T,V] or [B,V]
        # We'll assume HRM can handle (src,tgt) training if implemented. If not, fallback to wrapper.
        try:
            params = list(model.parameters())
            print(f"[model] HRM params: {sum(p.numel() for p in params)}")
        except Exception:
            print("[model] Warning: HRM appears not trainable, falling back to Seq2Seq wrapper.")
            trainable_hrm = False

    if not trainable_hrm:
        print("[model] Using fallback SimpleSeq2Seq wrapper.")
        vocab_size = len(tokenizer.vocab)
        model = SimpleSeq2Seq(vocab_size=vocab_size, embed_dim=128, enc_hid=256, dec_hid=256, pad_id=0).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # ignore PAD
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_loop(model, train_dl, optimizer, device, scaler=scaler, loss_fn=loss_fn)
        val_loss = eval_loop(model, val_dl, device, loss_fn=loss_fn)
        dt = time.time() - t0
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} time={dt:.1f}s")
        # checkpoint
        ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch}.pt")
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}, ckpt_path)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best.pt"))

    print("[done] Training finished. Best val loss:", best_val)
    print("Saved to", args.save_dir)

if __name__ == "__main__":
    main()
    