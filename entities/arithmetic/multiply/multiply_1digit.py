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
# Multiply1DigitEntity (BATCH-READY)
# -------------------------------------------------------------------
class Multiply1DigitEntity(EntityBase, nn.Module):
    """
    Neural Entity for single-digit multiplication:

        (a × b + carry_in) % 10

    Properties:
    - Predicts ONLY the product digit
    - Accepts FULL carry_in ∈ [0..9]
    - Does NOT predict carry_out
    - Safe for symbolic N-digit composition
    - Supports batched execution
    """

    name = "Multiply1DigitEntity"

    requires = [
        "arithmetic.digits_a",
        "arithmetic.digits_b",
        "arithmetic.carry",
    ]

    produces = [
        "arithmetic.prod",
    ]

    invariants = [
        "arithmetic.digits_a",
        "arithmetic.digits_b",
        "arithmetic.carry",
    ]

    supports_gpu = True
    deterministic = True

    def __init__(self, d_model=64):
        nn.Module.__init__(self)
        EntityBase.__init__(self)

        # --------------------------------------------------
        # Embeddings (FULL domain)
        # --------------------------------------------------
        self.embed_digit = nn.Embedding(10, d_model)
        self.embed_carry = nn.Embedding(10, d_model)

        # --------------------------------------------------
        # Digit prediction head
        # --------------------------------------------------
        self.transition = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 10),
        )

        # Optional MOR stop head (research-only)
        self.stop_head = nn.Sequential(
            nn.Linear(d_model * 3, 1),
            nn.Sigmoid(),
        )

        self.param_count = sum(p.numel() for p in self.parameters())

    # -----------------------------------------------------
    # Single-state execution (unchanged behavior)
    # -----------------------------------------------------
    def forward(
        self,
        state: GlobalState,
        max_steps: int = 3,
        stop_threshold: float = 0.6,
    ) -> None:

        device = self.embed_digit.weight.device

        a = state.arithmetic.digits_a[0]
        b = state.arithmetic.digits_b[0]
        carry_in = state.arithmetic.carry

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

        stop_probs = []

        for step in range(max_steps):
            logits = self.transition(h)

            state.arithmetic.prod_logits = logits
            state.arithmetic.prod = logits.argmax(dim=-1).item()

            stop_prob = self.stop_head(h).item()
            stop_probs.append(stop_prob)

            if stop_prob > stop_threshold:
                break

        state.trace.mor_steps = step + 1
        state.trace.avg_stop_prob = sum(stop_probs) / len(stop_probs)

    # -----------------------------------------------------
    # Batched execution (NEW — FAST)
    # -----------------------------------------------------
    def forward_batch(self, states: list[GlobalState]) -> None:
        """
        Batched digit inference.
        This is where GPU speedups happen.
        """

        device = self.embed_digit.weight.device
        batch_size = len(states)

        a = torch.tensor(
            [s.arithmetic.digits_a[0] for s in states],
            dtype=torch.long,
            device=device,
        )
        b = torch.tensor(
            [s.arithmetic.digits_b[0] for s in states],
            dtype=torch.long,
            device=device,
        )
        carry = torch.tensor(
            [s.arithmetic.carry for s in states],
            dtype=torch.long,
            device=device,
        )

        h = torch.cat(
            [
                self.embed_digit(a),
                self.embed_digit(b),
                self.embed_carry(carry),
            ],
            dim=-1,
        )

        logits = self.transition(h)
        preds = logits.argmax(dim=-1)

        for i, state in enumerate(states):
            state.arithmetic.prod_logits = logits[i : i + 1]
            state.arithmetic.prod = preds[i].item()
            state.trace.mor_steps = 1
            state.trace.avg_stop_prob = 1.0


# -------------------------------------------------------------------
# Dataset loader
# -------------------------------------------------------------------
def load_multiply1digit_jsonl(path):
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
def train_multiply1digit_entity(
    model: Multiply1DigitEntity,
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

        losses, entropies, grad_norms = [], [], []

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

            logits = state.arithmetic.prod_logits
            loss = F.cross_entropy(logits, target)

            optimizer.zero_grad()
            loss.backward()
            grad_norms.append(grad_global_norm(model))
            optimizer.step()

            losses.append(loss.item())
            entropies.append(tensor_entropy(logits))

        print(
            f"[Epoch {epoch+1}] "
            f"Loss={sum(losses)/len(losses):.6f} | "
            f"Entropy={sum(entropies)/len(entropies):.4f}"
        )

        epoch_metrics.append({
            "epoch": epoch + 1,
            "loss_mean": sum(losses) / len(losses),
            "entropy": sum(entropies) / len(entropies),
            "grad_norm": sum(grad_norms) / len(grad_norms),
        })

    if return_metrics:
        return epoch_metrics


# -------------------------------------------------------------------
# Evaluation (FULL DOMAIN)
# -------------------------------------------------------------------
def evaluate_multiply1digit_entity(model: Multiply1DigitEntity):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for a in range(10):
            for b in range(10):
                for carry_in in range(10):
                    state = GlobalState()
                    state.arithmetic.digits_a = [a]
                    state.arithmetic.digits_b = [b]
                    state.arithmetic.carry = carry_in

                    model.forward(state)

                    if state.arithmetic.prod == (a * b + carry_in) % 10:
                        correct += 1
                    total += 1

    print(
        f"Multiply1DigitEntity accuracy: "
        f"{correct}/{total} = {correct/total:.3f}"
    )