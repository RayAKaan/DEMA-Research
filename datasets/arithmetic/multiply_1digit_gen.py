import json
import os

# --------------------------------------------------
# Project root (absolute, strict)
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../..")
)

BASE_DIR = os.path.join(
    PROJECT_ROOT,
    "DEMA-RESEARCH",
    "mega-math-model",
    "data",
    "synthetic",
    "arithmetic",
)

PROD_PATH = os.path.join(BASE_DIR, "multiply_1digit.jsonl")
CARRY_PATH = os.path.join(BASE_DIR, "mul_carry.jsonl")

# --------------------------------------------------
# STRICT path checks (no folder creation)
# --------------------------------------------------
if not os.path.isdir(BASE_DIR):
    raise RuntimeError(
        f"Target directory does NOT exist:\n{BASE_DIR}"
    )

# --------------------------------------------------
# Dataset generation (FULL ALGEBRAIC CLOSURE)
# --------------------------------------------------
def generate_datasets():
    prod_count = 0
    carry_count = 0

    with open(PROD_PATH, "w") as f_prod, open(CARRY_PATH, "w") as f_carry:
        for a in range(10):
            for b in range(10):
                for carry_in in range(10):  # FULL DOMAIN

                    raw = a * b + carry_in

                    # -----------------------------
                    # Multiply1DigitEntity dataset
                    # -----------------------------
                    prod_sample = {
                        "a": a,
                        "b": b,
                        "carry_in": carry_in,
                        "target": raw % 10,
                    }
                    f_prod.write(json.dumps(prod_sample) + "\n")
                    prod_count += 1

                    # -----------------------------
                    # MulCarryEntity dataset
                    # -----------------------------
                    carry_sample = {
                        "a": a,
                        "b": b,
                        "carry_in": carry_in,
                        "target": raw // 10,
                    }
                    f_carry.write(json.dumps(carry_sample) + "\n")
                    carry_count += 1

    print("Generated algebraically complete datasets:")
    print(f" - Multiply1DigitEntity : {prod_count} samples")
    print(f" - MulCarryEntity       : {carry_count} samples")
    print(f"\nPaths:")
    print(f" - {PROD_PATH}")
    print(f" - {CARRY_PATH}")

# --------------------------------------------------
# Entry point
# --------------------------------------------------
if __name__ == "__main__":
    generate_datasets()