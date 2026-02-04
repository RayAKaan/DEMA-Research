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

CARRY_PATH = os.path.join(BASE_DIR, "mul_carry.jsonl")

# --------------------------------------------------
# STRICT path checks (no folder creation)
# --------------------------------------------------
if not os.path.isdir(BASE_DIR):
    raise RuntimeError(
        f"Target directory does NOT exist:\n{BASE_DIR}"
    )

# --------------------------------------------------
# Dataset generation (EXHAUSTIVE)
# --------------------------------------------------
def generate_mulcarry_dataset():
    count = 0

    with open(CARRY_PATH, "w") as f:
        for a in range(10):
            for b in range(10):
                for carry_in in range(9):  # ← CRITICAL FIX
                    raw = a * b + carry_in
                    carry_out = raw // 10

                    f.write(json.dumps({
                        "a": a,
                        "b": b,
                        "carry_in": carry_in,
                        "target": carry_out,
                    }) + "\n")

                    count += 1

    print("Generated MulCarry dataset:")
    print(f" - Path   : {CARRY_PATH}")
    print(f" - Samples: {count} (expected 900)")

    assert count == 900, "Dataset size mismatch!"

# --------------------------------------------------
# Entry point
# --------------------------------------------------
if __name__ == "__main__":
    generate_mulcarry_dataset()