import os
import torch

from entities.arithmetic.multiply.multiply_1digit import (
    Multiply1DigitEntity,
    load_multiply1digit_jsonl,
    train_multiply1digit_entity,
    evaluate_multiply1digit_entity,
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


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    # --------------------------------------------------
    # Config (STABLE)
    # --------------------------------------------------
    MODEL_DIR = "models/entities/arithmetic/multiply/multiply_1digit"
    DATASET_PATH = "data/synthetic/arithmetic/multiply_1digit.jsonl"
    LOG_PATH = "logs/multiply/multiply_1digit_metrics.csv"

    d_model = 64
    epochs = 40
    lr = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------
    # Prepare
    # --------------------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    model = Multiply1DigitEntity(d_model=d_model)
    dataset = load_multiply1digit_jsonl(DATASET_PATH)

    total_params, trainable_params = count_parameters(model)
    device_info = get_device_info(device)

    # --------------------------------------------------
    # CSV Logger
    # --------------------------------------------------
    logger = CSVLogger(
        path=LOG_PATH,
        fieldnames=[
            # run-level metadata
            "model",
            "device",
            "gpu_name",
            "d_model",
            "learning_rate",
            "epochs",
            "total_params",
            "trainable_params",
            # epoch-level metrics
            "epoch",
            "loss_mean",
            "loss_std",
            "entropy",
            "grad_norm",
            "mor_steps",
            "stop_prob",
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
        "epochs": epochs,
        "total_params": total_params,
        "trainable_params": trainable_params,
    })

    # --------------------------------------------------
    # Train
    # --------------------------------------------------
    print("\n=== Training Multiply1DigitEntity ===\n")

    metrics_per_epoch = train_multiply1digit_entity(
        model,
        dataset,
        epochs=epochs,
        lr=lr,
        device=device,
        return_metrics=True,
    )

    for m in metrics_per_epoch:
        logger.log(m)

    logger.close()

    # --------------------------------------------------
    # Evaluate (FULL carry domain 0..9)
    # --------------------------------------------------
    print("\n=== Evaluating Multiply1DigitEntity ===\n")
    evaluate_multiply1digit_entity(model)

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    model_path = os.path.join(MODEL_DIR, "model.pt")
    torch.save(model.state_dict(), model_path)

    print(f"\nModel saved to: {model_path}")
    print(f"Metrics logged to: {LOG_PATH}\n")


if __name__ == "__main__":
    main()