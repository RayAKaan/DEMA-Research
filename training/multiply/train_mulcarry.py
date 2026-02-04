import os
import sys

# --------------------------------------------------
# Ensure project root is on PYTHONPATH
# --------------------------------------------------
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(PROJECT_ROOT)

import torch

from entities.arithmetic.multiply.mul_carry import (
    MulCarryEntity,
    load_mulcarry_jsonl,
    train_mulcarry_entity,
    evaluate_mulcarry_entity,
)

from training.utils.metrics_logger import CSVLogger


# --------------------------------------------------
# Utility helpers
# --------------------------------------------------
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_device_info(device):
    if device.startswith("cuda") and torch.cuda.is_available():
        return {
            "device": "cuda",
            "gpu_name": torch.cuda.get_device_name(0),
        }
    return {
        "device": "cpu",
        "gpu_name": "N/A",
    }


def main():
    # --------------------------------------------------
    # Config (STABLE)
    # --------------------------------------------------
    MODEL_DIR = "models/entities/arithmetic/multiply/mul_carry"
    DATASET_PATH = "data/synthetic/arithmetic/mul_carry.jsonl"
    LOG_PATH = "logs/multiply/mul_carry_metrics.csv"

    d_model = 32
    max_epochs = 60
    lr = 1e-3               # stable base LR
    lr_step = 10
    lr_gamma = 0.3          # decay factor

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------
    # Prepare
    # --------------------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    model = MulCarryEntity(d_model=d_model)
    dataset = load_mulcarry_jsonl(DATASET_PATH)

    total_params, trainable_params = count_parameters(model)
    device_info = get_device_info(device)

    # --------------------------------------------------
    # CSV Logger
    # --------------------------------------------------
    logger = CSVLogger(
        path=LOG_PATH,
        fieldnames=[
            "model",
            "device",
            "gpu_name",
            "d_model",
            "learning_rate",
            "epochs",
            "total_params",
            "trainable_params",
            "epoch",
            "loss_mean",
            "grad_norm",
        ],
    )

    # --------------------------------------------------
    # Log run metadata (once)
    # --------------------------------------------------
    logger.log({
        "model": model.name,
        "device": device_info["device"],
        "gpu_name": device_info["gpu_name"],
        "d_model": d_model,
        "learning_rate": lr,
        "epochs": max_epochs,
        "total_params": total_params,
        "trainable_params": trainable_params,
    })

    # --------------------------------------------------
    # Optimizer + Scheduler
    # --------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=lr_step,
        gamma=lr_gamma,
    )

    # --------------------------------------------------
    # Training (NO early stopping)
    # --------------------------------------------------
    print("\n=== Training MulCarryEntity (deterministic) ===\n")

    model.to(device)
    model.train()

    metrics = []

    for epoch in range(1, max_epochs + 1):
        losses = []
        grad_norms = []

        for sample in dataset:
            from state import GlobalState

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

            loss = torch.nn.functional.cross_entropy(
                state.arithmetic.carry_logits,
                target,
            )

            optimizer.zero_grad()
            loss.backward()

            grad_norm = sum(
                p.grad.norm().item()
                for p in model.parameters()
                if p.grad is not None
            )

            optimizer.step()

            losses.append(loss.item())
            grad_norms.append(grad_norm)

        scheduler.step()

        mean_loss = sum(losses) / len(losses)
        mean_grad = sum(grad_norms) / len(grad_norms)

        print(
            f"[Epoch {epoch:02d}] "
            f"Loss={mean_loss:.6f} | "
            f"GradNorm={mean_grad:.4f} | "
            f"LR={scheduler.get_last_lr()[0]:.2e}"
        )

        metrics.append({
            "epoch": epoch,
            "loss_mean": mean_loss,
            "grad_norm": mean_grad,
        })

        logger.log(metrics[-1])

    logger.close()

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    print("\n=== Evaluating MulCarryEntity ===\n")
    evaluate_mulcarry_entity(model)

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------
    model_path = os.path.join(MODEL_DIR, "model.pt")
    torch.save(model.state_dict(), model_path)

    print(f"\n✅ Model saved to: {model_path}")
    print(f"📊 Metrics logged to: {LOG_PATH}\n")


if __name__ == "__main__":
    main()