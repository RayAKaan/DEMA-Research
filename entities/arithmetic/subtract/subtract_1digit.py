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
# Subtract1DigitEntity (BORROW-AWARE, FINAL)
# -------------------------------------------------------------------
class Subtract1DigitEntity(EntityBase, nn.Module):
    """
    Neural Entity for single-digit subtraction:

        (a - b - borrow_in) % 10

    - Receives borrow_in ∈ {0,1}
    - Predicts ONLY the digit
    - Borrow handled by BorrowEntity
    """

    name = "Subtract1DigitEntity"

    requires = [
        "arithmetic.digits_a",
        "arithmetic.digits_b",
        "arithmetic.borrow",
    ]

    produces = [
        "arithmetic.diff",
    ]

    invariants = [
        "arithmetic.digits_a",
        "arithmetic.digits_b",
        "arithmetic.borrow",
    ]

    def __init__(self, d_model=64):
        nn.Module.__init__(self)
        EntityBase.__init__(self)

        self.embed_digit = nn.Embedding(10, d_model)
        self.embed_borrow = nn.Embedding(2, d_model)

        self.transition = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 10),
        )

        # MOR head (kept for research consistency)
        self.stop_head = nn.Sequential(
            nn.Linear(d_model * 3, 1),
            nn.Sigmoid(),
        )

    # -----------------------------------------------------
    # Forward
    # -----------------------------------------------------
    def forward(
        self,
        state: GlobalState,
        max_steps: int = 3,
        stop_threshold: float = 0.6,
    ) -> None:

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

        stop_probs = []

        for step in range(max_steps):
            logits = self.transition(h)

            state.arithmetic.diff_logits = logits
            state.arithmetic.diff = logits.argmax(dim=-1).item()

            stop_prob = self.stop_head(h).item()
            stop_probs.append(stop_prob)

            state.trace.decisions.append(
                f"Subtract1DigitEntity step={step+1} stop_prob={stop_prob:.3f}"
            )

            if stop_prob > stop_threshold:
                break

        state.trace.mor_steps = step + 1
        state.trace.avg_stop_prob = sum(stop_probs) / len(stop_probs)


# -------------------------------------------------------------------
# Dataset loader (MATCHES YOUR DATASET)
# -------------------------------------------------------------------
def load_subtract1digit_jsonl(path):
    """
    JSONL format:
    {
        "a": int,
        "b": int,
        "borrow_in": 0 or 1,
        "target": (a - b - borrow_in) % 10
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
                "target": obj["target"],
            })
    return data


# -------------------------------------------------------------------
# Training loop (LOGGING FIXED)
# -------------------------------------------------------------------
def train_subtract1digit_entity(
    model: Subtract1DigitEntity,
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

        losses, entropies, grad_norms, mor_steps, stop_probs = [], [], [], [], []

        for sample in dataset:
            state = GlobalState()
            state.arithmetic.digits_a = [sample["a"]]
            state.arithmetic.digits_b = [sample["b"]]
            state.arithmetic.borrow = sample["borrow_in"]

            model.forward(state)

            target = torch.tensor(
                [sample["target"]],
                device=device,
                dtype=torch.long,
            )

            logits = state.arithmetic.diff_logits
            loss = F.cross_entropy(logits, target)

            optimizer.zero_grad()
            loss.backward()
            grad_norms.append(grad_global_norm(model))
            optimizer.step()

            losses.append(loss.item())
            entropies.append(tensor_entropy(logits))
            mor_steps.append(state.trace.mor_steps)
            stop_probs.append(state.trace.avg_stop_prob)

        mean_loss = sum(losses) / len(losses)
        std_loss = math.sqrt(sum((l - mean_loss) ** 2 for l in losses) / len(losses))
        avg_grad = sum(grad_norms) / len(grad_norms)

        print(
            f"[Epoch {epoch+1}] "
            f"Loss={mean_loss:.6f} ± {std_loss:.6f} | "
            f"GradNorm={avg_grad:.4f}"
        )

        epoch_metrics.append({
            "epoch": epoch + 1,
            "loss_mean": mean_loss,
            "loss_std": std_loss,
            "grad_norm": avg_grad,   # ✅ FIX
            "mor_steps": sum(mor_steps) / len(mor_steps),
            "stop_prob": sum(stop_probs) / len(stop_probs),
        })

    if return_metrics:
        return epoch_metrics


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------
def evaluate_subtract1digit_entity(model: Subtract1DigitEntity):
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

                    expected = (a - b - borrow_in) % 10
                    correct += int(state.arithmetic.diff == expected)
                    total += 1

    print(
        f"Subtract1DigitEntity accuracy: "
        f"{correct}/{total} = {correct/total:.3f}"
    )