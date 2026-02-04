import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import random

from entities.base import EntityBase
from state import GlobalState


# -------------------------------------------------------------------
# Utility
# -------------------------------------------------------------------
def grad_global_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.norm().item()
    return total


# -------------------------------------------------------------------
# MulCarryEntity
# -------------------------------------------------------------------
class MulCarryEntity(EntityBase, nn.Module):
    """
    Neural entity that predicts carry_out ∈ {0..8}
    for single-digit multiplication.

    raw = a * b + carry_in
    carry_out = raw // 10
    """

    name = "MulCarryEntity"

    requires = [
        "arithmetic.digits_a",
        "arithmetic.digits_b",
        "arithmetic.carry",
    ]

    produces = [
        "arithmetic.carry",
    ]

    invariants = [
        "arithmetic.digits_a",
        "arithmetic.digits_b",
    ]

    def __init__(self, d_model=32):
        nn.Module.__init__(self)
        EntityBase.__init__(self)

        # digits: 0–9
        self.embed_digit = nn.Embedding(10, d_model)

        # carry_in: 0–8  ✅ CRITICAL FIX
        self.embed_carry = nn.Embedding(9, d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 9),  # carry_out ∈ [0..8]
        )

    # -----------------------------------------------------
    # Entity execution
    # -----------------------------------------------------
    def forward(self, state: GlobalState) -> None:
        a = state.arithmetic.digits_a[0]
        b = state.arithmetic.digits_b[0]
        carry_in = state.arithmetic.carry

        # Safety (keeps research honest, avoids silent corruption)
        assert 0 <= a <= 9
        assert 0 <= b <= 9
        assert 0 <= carry_in <= 8

        device = self.embed_digit.weight.device

        a_t = torch.tensor([a], dtype=torch.long, device=device)
        b_t = torch.tensor([b], dtype=torch.long, device=device)
        c_t = torch.tensor([carry_in], dtype=torch.long, device=device)

        h = torch.cat(
            [
                self.embed_digit(a_t),
                self.embed_digit(b_t),
                self.embed_carry(c_t),
            ],
            dim=-1,
        )

        logits = self.classifier(h)
        carry_out = logits.argmax(dim=-1).item()

        state.arithmetic.carry_logits = logits
        state.arithmetic.carry = carry_out

        state.trace.decisions.append(
            f"MulCarryEntity: ({a}×{b} + {carry_in}) → carry={carry_out}"
        )


# -------------------------------------------------------------------
# Dataset loader
# -------------------------------------------------------------------
def load_mulcarry_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            data.append({
                "a": obj["a"],
                "b": obj["b"],
                "carry_in": obj["carry_in"],
                "target": obj["target"],
            })
    return data


# -------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------
def train_mulcarry_entity(
    model: MulCarryEntity,
    dataset,
    epochs=10,
    lr=1e-3,
    device=None,
    return_metrics=False,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    epoch_metrics = []

    for epoch in range(epochs):
        random.shuffle(dataset)
        losses = []
        grad_norms = []

        for sample in dataset:
            state = GlobalState()
            state.arithmetic.digits_a = [sample["a"]]
            state.arithmetic.digits_b = [sample["b"]]
            state.arithmetic.carry = sample["carry_in"]

            model.forward(state)

            target = torch.tensor(
                [sample["target"]],
                dtype=torch.long,
                device=device,
            )

            logits = state.arithmetic.carry_logits
            loss = F.cross_entropy(logits, target)

            optimizer.zero_grad()
            loss.backward()
            grad_norms.append(grad_global_norm(model))
            optimizer.step()

            losses.append(loss.item())

        mean_loss = sum(losses) / len(losses)
        avg_grad = sum(grad_norms) / len(grad_norms)

        print(
            f"[Epoch {epoch+1}] "
            f"Loss={mean_loss:.6f} | GradNorm={avg_grad:.4f}"
        )

        epoch_metrics.append({
            "epoch": epoch + 1,
            "loss_mean": mean_loss,
            "grad_norm": avg_grad,
        })

    if return_metrics:
        return epoch_metrics


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------
def evaluate_mulcarry_entity(model: MulCarryEntity):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for a in range(10):
            for b in range(10):
                for carry_in in range(9):
                    state = GlobalState()
                    state.arithmetic.digits_a = [a]
                    state.arithmetic.digits_b = [b]
                    state.arithmetic.carry = carry_in

                    model.forward(state)

                    expected = (a * b + carry_in) // 10
                    if state.arithmetic.carry == expected:
                        correct += 1
                    total += 1

    print(
        f"MulCarryEntity accuracy: "
        f"{correct}/{total} = {correct/total:.4f}"
    )