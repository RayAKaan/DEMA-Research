import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import json

from entities.base import EntityBase
from state import GlobalState


# -------------------------------------------------------------------
# Utility metrics
# -------------------------------------------------------------------
def grad_global_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.norm().item()
    return total


def binary_entropy_from_logits(logits):
    """
    Binary entropy for borrow_out prediction.
    """
    probs = torch.softmax(logits, dim=-1)
    p = probs[:, 1]
    return -(
        p * torch.log(p + 1e-9) +
        (1 - p) * torch.log(1 - p + 1e-9)
    ).mean().item()


# -------------------------------------------------------------------
# BorrowEntity
# -------------------------------------------------------------------
class BorrowEntity(EntityBase, nn.Module):
    """
    Neural Entity that predicts borrow_out ∈ {0,1}
    for single-digit subtraction.

    borrow_out = 1 if (a - borrow_in) < b
    """

    name = "BorrowEntity"

    requires = [
        "arithmetic.digits_a",
        "arithmetic.digits_b",
        "arithmetic.borrow",
    ]

    produces = [
        "arithmetic.borrow",
    ]

    invariants = [
        "arithmetic.digits_a",
        "arithmetic.digits_b",
    ]

    def __init__(self, d_model=32):
        nn.Module.__init__(self)
        EntityBase.__init__(self)

        self.embed_digit = nn.Embedding(10, d_model)
        self.embed_borrow = nn.Embedding(2, d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),  # binary classification
        )

    # -----------------------------------------------------
    # Entity execution
    # -----------------------------------------------------
    def forward(self, state: GlobalState) -> None:
        a = state.arithmetic.digits_a[0]
        b = state.arithmetic.digits_b[0]
        borrow_in = state.arithmetic.borrow

        device = self.embed_digit.weight.device

        a_t = torch.tensor([a], dtype=torch.long, device=device)
        b_t = torch.tensor([b], dtype=torch.long, device=device)
        bi_t = torch.tensor([borrow_in], dtype=torch.long, device=device)

        h = torch.cat(
            [
                self.embed_digit(a_t),
                self.embed_digit(b_t),
                self.embed_borrow(bi_t),
            ],
            dim=-1,
        )

        logits = self.classifier(h)

        borrow_out = logits.argmax(dim=-1).item()

        state.arithmetic.borrow_logits = logits
        state.arithmetic.borrow = borrow_out

        state.trace.decisions.append(
            f"BorrowEntity borrow_in={borrow_in} → borrow_out={borrow_out}"
        )


# -------------------------------------------------------------------
# Dataset loader
# -------------------------------------------------------------------
def load_borrow_jsonl(path):
    """
    Expected JSONL format:
    {
      "a": int,
      "b": int,
      "borrow_in": 0 or 1,
      "borrow_out": 0 or 1
    }
    """
    data = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            data.append({
                "a": obj["a"],
                "b": obj["b"],
                "borrow_in": obj["borrow_in"],
                "target": obj["borrow_out"],
            })
    return data


# -------------------------------------------------------------------
# Training loop (research-aligned)
# -------------------------------------------------------------------
def train_borrow_entity(
    model: BorrowEntity,
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
        entropies = []

        for sample in dataset:
            state = GlobalState()
            state.arithmetic.digits_a = [sample["a"]]
            state.arithmetic.digits_b = [sample["b"]]
            state.arithmetic.borrow = sample["borrow_in"]

            model.forward(state)

            target = torch.tensor(
                [sample["target"]],
                dtype=torch.long,
                device=device,
            )

            logits = state.arithmetic.borrow_logits
            loss = F.cross_entropy(logits, target)

            optimizer.zero_grad()
            loss.backward()
            grad_norms.append(grad_global_norm(model))
            optimizer.step()

            losses.append(loss.item())
            entropies.append(binary_entropy_from_logits(logits))

        mean_loss = sum(losses) / len(losses)
        std_loss = math.sqrt(
            sum((l - mean_loss) ** 2 for l in losses) / len(losses)
        )

        avg_grad = sum(grad_norms) / len(grad_norms)
        avg_entropy = sum(entropies) / len(entropies)

        print(
            f"[Epoch {epoch+1}] "
            f"Loss={mean_loss:.6f} ± {std_loss:.6f} | "
            f"Entropy={avg_entropy:.4f} | "
            f"GradNorm={avg_grad:.4f}"
        )

        epoch_metrics.append({
            "epoch": epoch + 1,
            "loss_mean": mean_loss,
            "loss_std": std_loss,
            "entropy": avg_entropy,
            "grad_norm": avg_grad,
            "mor_steps": 1.0,   # single-step entity
            "stop_prob": 1.0,   # always halts
        })

    if return_metrics:
        return epoch_metrics


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------
def evaluate_borrow_entity(model: BorrowEntity):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for a in range(10):
            for b in range(10):
                for borrow_in in [0, 1]:
                    state = GlobalState()
                    state.arithmetic.digits_a = [a]
                    state.arithmetic.digits_b = [b]
                    state.arithmetic.borrow = borrow_in

                    model.forward(state)

                    expected = 1 if (a - borrow_in) < b else 0
                    if state.arithmetic.borrow == expected:
                        correct += 1
                    total += 1

    print(
        f"BorrowEntity accuracy: "
        f"{correct}/{total} = {correct/total:.3f}"
    )