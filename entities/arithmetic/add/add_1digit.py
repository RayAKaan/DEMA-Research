import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import json
import math

from entities.base import EntityBase
from state import GlobalState


# -------------------------------------------------------------------
# Utility metrics
# -------------------------------------------------------------------
def tensor_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    return -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean().item()


def grad_global_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.norm().item()
    return total


# -------------------------------------------------------------------
# Add1DigitEntity (CORRECT)
# -------------------------------------------------------------------
class Add1DigitEntity(EntityBase, nn.Module):
    """
    Neural Entity for single-digit addition:

        (a + b) % 10

    - NO carry input
    - Predicts ONLY sum digit
    - Carry is handled by CarryEntity
    """

    name = "Add1DigitEntity"

    requires = [
        "arithmetic.digits_a",
        "arithmetic.digits_b",
    ]

    produces = [
        "arithmetic.sum",
    ]

    invariants = [
        "arithmetic.digits_a",
        "arithmetic.digits_b",
    ]

    def __init__(self, d_model=64):
        nn.Module.__init__(self)
        EntityBase.__init__(self)

        self.embed = nn.Embedding(10, d_model)

        self.transition = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 10),
        )

    # -----------------------------------------------------
    # Entity execution
    # -----------------------------------------------------
    def forward(self, state: GlobalState) -> None:
        a = state.arithmetic.digits_a[0]
        b = state.arithmetic.digits_b[0]

        device = self.embed.weight.device

        a_t = torch.tensor([a], dtype=torch.long, device=device)
        b_t = torch.tensor([b], dtype=torch.long, device=device)

        h = torch.cat(
            [self.embed(a_t), self.embed(b_t)],
            dim=-1,
        )

        logits = self.transition(h)

        state.arithmetic.sum_logits = logits
        state.arithmetic.sum = logits.argmax(dim=-1).item()

        state.trace.decisions.append(
            f"Add1DigitEntity a={a}, b={b} → sum={state.arithmetic.sum}"
        )


# -------------------------------------------------------------------
# Dataset loader (MATCHES YOUR DATASET)
# -------------------------------------------------------------------
def load_add1digit_jsonl(path):
    """
    Expected JSONL format:
        {"a": int, "b": int, "target": int}
    """
    data = []

    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            data.append({
                "a": obj["a"],
                "b": obj["b"],
                "target": obj["target"],
            })

    return data


# -------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------
def train_add1digit_entity(
    model: Add1DigitEntity,
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
        entropies = []
        grad_norms = []

        for sample in dataset:
            state = GlobalState()
            state.arithmetic.digits_a = [sample["a"]]
            state.arithmetic.digits_b = [sample["b"]]

            model.forward(state)

            target = torch.tensor(
                [sample["target"]],
                device=device,
                dtype=torch.long,
            )

            logits = state.arithmetic.sum_logits
            loss = F.cross_entropy(logits, target)

            optimizer.zero_grad()
            loss.backward()
            grad_norms.append(grad_global_norm(model))
            optimizer.step()

            losses.append(loss.item())
            entropies.append(tensor_entropy(logits))

        mean_loss = sum(losses) / len(losses)

        print(
            f"[Epoch {epoch+1}] "
            f"Loss={mean_loss:.6f} | "
            f"Entropy={sum(entropies)/len(entropies):.4f} | "
            f"GradNorm={sum(grad_norms)/len(grad_norms):.4f}"
        )

        epoch_metrics.append({
            "epoch": epoch + 1,
            "loss_mean": mean_loss,
        })

    if return_metrics:
        return epoch_metrics


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------
def evaluate_add1digit_entity(model: Add1DigitEntity):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for a in range(10):
            for b in range(10):
                state = GlobalState()
                state.arithmetic.digits_a = [a]
                state.arithmetic.digits_b = [b]

                model.forward(state)

                expected = (a + b) % 10
                if state.arithmetic.sum == expected:
                    correct += 1
                total += 1

    print(
        f"Add1DigitEntity accuracy: "
        f"{correct}/{total} = {correct/total:.3f}"
    )