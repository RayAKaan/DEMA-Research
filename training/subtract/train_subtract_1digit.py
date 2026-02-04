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

from entities.arithmetic.subtract.subtract_1digit import (
    Subtract1DigitEntity,
    train_subtract1digit_entity,
    evaluate_subtract1digit_entity,
)

from training.utils.metrics_logger import CSVLogger
import json


# --------------------------------------------------
# Dataset loader (MATCHES {a,b,borrow_in,target})
# --------------------------------------------------
def load_subtract1digit_jsonl(path):
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


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    MODEL_DIR = "models/entities/arithmetic/subtract/subtract_1digit"
    DATASET_PATH = "data/synthetic/arithmetic/subtract_1digit.jsonl"
    LOG_PATH = "logs/subtract/subtract_1digit_metrics.csv"

    d_model = 64
    max_epochs = 10
    lr = 1e-3

    LOSS_EPS = 1e-4
    GRAD_EPS = 1e-3
    PATIENCE = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------
    # Prepare
    # --------------------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    model = Subtract1DigitEntity(d_model=d_model)
    dataset = load_subtract1digit_jsonl(DATASET_PATH)

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
            "loss_std",
            "entropy",
            "grad_norm",
            "mor_steps",
            "stop_prob",
        ],
    )

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
    # Train with early stopping
    # --------------------------------------------------
    print("\n=== Training Subtract1DigitEntity ===\n")

    best_loss = float("inf")
    best_epoch = None
    best_state = None
    patience_counter = 0

    metrics = train_subtract1digit_entity(
        model,
        dataset,
        epochs=max_epochs,
        lr=lr,
        device=device,
        return_metrics=True,
    )

    for m in metrics:
        logger.log(m)

        loss = m["loss_mean"]
        grad = m["grad_norm"]

        if loss < best_loss:
            best_loss = loss
            best_epoch = m["epoch"]
            best_state = {
                k: v.detach().cpu()
                for k, v in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if loss < LOSS_EPS and grad < GRAD_EPS:
            print(
                f"\n🟢 Converged at epoch {m['epoch']} "
                f"(loss={loss:.2e}, grad={grad:.2e})"
            )
            break

        if patience_counter > PATIENCE:
            print(
                f"\n🟡 Early stopping at epoch {m['epoch']} "
                f"(no improvement for {PATIENCE} epoch)"
            )
            break

    logger.close()

    # --------------------------------------------------
    # Restore best model
    # --------------------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n✅ Restored best model from epoch {best_epoch}")

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------
    print("\n=== Evaluating Subtract1DigitEntity ===\n")
    evaluate_subtract1digit_entity(model)

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    model_path = os.path.join(MODEL_DIR, "model.pt")
    torch.save(model.state_dict(), model_path)

    print(f"\nModel saved to: {model_path}")
    print(f"Metrics logged to: {LOG_PATH}\n")


if __name__ == "__main__":
    main()