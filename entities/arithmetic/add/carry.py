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
# CarryEntity
# -------------------------------------------------------------------
class CarryEntity(EntityBase, nn.Module):
    """
    Neural Entity that predicts carry-out for digit addition.

    Learns:
        (digit_a, digit_b, carry_in) → carry_out ∈ {0,1}
    """

    name = "CarryEntity"

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

    def __init__(self, d_model=64):
        nn.Module.__init__(self)
        EntityBase.__init__(self)

        self.embed_digit = nn.Embedding(10, d_model)
        self.embed_carry = nn.Embedding(2, d_model)

        self.transition = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2)
        )

    # -----------------------------------------------------
    # Entity execution
    # -----------------------------------------------------
    def forward(self, state: GlobalState) -> None:
        a = state.arithmetic.digits_a[0]
        b = state.arithmetic.digits_b[0]
        carry_in = state.arithmetic.carry

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

        logits = self.transition(h)

        state.arithmetic.carry_logits = logits
        state.arithmetic.carry = logits.argmax(dim=-1).item()

        state.trace.decisions.append(
            f"CarryEntity carry_in={carry_in} → carry_out={state.arithmetic.carry}"
        )


# -------------------------------------------------------------------
# Dataset loader
# -------------------------------------------------------------------
def load_carry_dataset(path):
    """
    From add_1digit.jsonl derive:
        a, b, carry_in ∈ {0,1}, target_carry
    """
    data = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            a, b = map(int, obj["input"].split("+"))
            total = int(obj["output"])

            for carry_in in [0, 1]:
                carry_out = 1 if (a + b + carry_in) >= 10 else 0
                data.append({
                    "a": a,
                    "b": b,
                    "carry_in": carry_in,
                    "target": carry_out,
                })
    return data


# -------------------------------------------------------------------
# Training loop (instrumented, CUDA-safe)
# -------------------------------------------------------------------
def train_carry_entity(
    model: CarryEntity,
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

        correct_0 = 0
        total_0 = 0
        correct_1 = 0
        total_1 = 0

        for sample in dataset:
            state = GlobalState()
            state.arithmetic.digits_a = [sample["a"]]
            state.arithmetic.digits_b = [sample["b"]]
            state.arithmetic.carry = sample["carry_in"]

            model.forward(state)

            target = torch.tensor(
                [sample["target"]],
                device=device,
                dtype=torch.long,
            )

            logits = state.arithmetic.carry_logits
            loss = F.cross_entropy(logits, target)

            optimizer.zero_grad()
            loss.backward()

            grad_norms.append(grad_global_norm(model))
            optimizer.step()

            losses.append(loss.item())
            entropies.append(tensor_entropy(logits))

            pred = logits.argmax(dim=-1).item()
            if sample["carry_in"] == 0:
                total_0 += 1
                correct_0 += int(pred == target.item())
            else:
                total_1 += 1
                correct_1 += int(pred == target.item())

        mean_loss = sum(losses) / len(losses)
        std_loss = math.sqrt(
            sum((l - mean_loss) ** 2 for l in losses) / len(losses)
        )

        avg_entropy = sum(entropies) / len(entropies)
        avg_grad = sum(grad_norms) / len(grad_norms)

        acc_0 = correct_0 / max(total_0, 1)
        acc_1 = correct_1 / max(total_1, 1)

        print(
            f"[Epoch {epoch+1}] "
            f"Loss={mean_loss:.6f} ± {std_loss:.6f} | "
            f"Entropy={avg_entropy:.4f} | "
            f"GradNorm={avg_grad:.4f} | "
            f"Acc(c0)={acc_0:.3f} | Acc(c1)={acc_1:.3f}"
        )

        epoch_metrics.append({
            "epoch": epoch + 1,
            "loss_mean": mean_loss,
            "loss_std": std_loss,
            "entropy": avg_entropy,
            "grad_norm": avg_grad,
            "acc_carry0": acc_0,
            "acc_carry1": acc_1,
        })

    if return_metrics:
        return epoch_metrics


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------
def evaluate_carry_entity(model: CarryEntity):
    model.eval()

    correct = 0
    total = 0
    correct_0 = 0
    total_0 = 0
    correct_1 = 0
    total_1 = 0

    with torch.no_grad():
        for a in range(10):
            for b in range(10):
                for carry_in in [0, 1]:
                    state = GlobalState()
                    state.arithmetic.digits_a = [a]
                    state.arithmetic.digits_b = [b]
                    state.arithmetic.carry = carry_in

                    model.forward(state)

                    expected = 1 if (a + b + carry_in) >= 10 else 0
                    pred = state.arithmetic.carry

                    total += 1
                    correct += int(pred == expected)

                    if carry_in == 0:
                        total_0 += 1
                        correct_0 += int(pred == expected)
                    else:
                        total_1 += 1
                        correct_1 += int(pred == expected)

    print(
        f"CarryEntity accuracy: {correct}/{total} = {correct/total:.3f}\n"
        f"  carry_in=0: {correct_0}/{total_0} = {correct_0/total_0:.3f}\n"
        f"  carry_in=1: {correct_1}/{total_1} = {correct_1/total_1:.3f}"
    )


# -------------------------------------------------------------------
# Standalone run
# -------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    entity = CarryEntity(d_model=64)

    dataset = load_carry_dataset(
        "data/synthetic/arithmetic/add_1digit.jsonl"
    )

    train_carry_entity(
        entity,
        dataset,
        epochs=10,
        lr=1e-3,
        device=device,
    )

    evaluate_carry_entity(entity)