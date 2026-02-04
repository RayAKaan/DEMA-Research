import os
import torch

from entities.arithmetic.add.carry import (
    CarryEntity,
    load_carry_dataset,
    train_carry_entity,
    evaluate_carry_entity,
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
    # Config
    # --------------------------------------------------
    MODEL_DIR = "models/entities/arithmetic/carry"
    DATASET_PATH = "data/synthetic/arithmetic/add_1digit.jsonl"
    LOG_PATH = "logs/carry_metrics.csv"

    d_model = 64
    epochs = 10
    lr = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------
    # Prepare
    # --------------------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    model = CarryEntity(d_model=d_model)
    dataset = load_carry_dataset(DATASET_PATH)

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
            "acc_carry0",
            "acc_carry1",
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
    print("\n=== Training CarryEntity ===\n")

    metrics_per_epoch = train_carry_entity(
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
    # Evaluate
    # --------------------------------------------------
    print("\n=== Evaluating CarryEntity ===\n")
    evaluate_carry_entity(model)

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    model_path = os.path.join(MODEL_DIR, "model.pt")
    torch.save(model.state_dict(), model_path)

    print(f"\nModel saved to: {model_path}")
    print(f"Metrics logged to: {LOG_PATH}\n")


if __name__ == "__main__":
    main()