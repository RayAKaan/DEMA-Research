import random
import json
import os


def generate_example():
    """
    Generates a single 1-digit addition example
    in structured numeric form.

    Learns:
        (a + b) % 10
    """
    a = random.randint(0, 9)
    b = random.randint(0, 9)

    total = a + b

    return {
        "a": a,
        "b": b,
        "target": total % 10,
    }


def generate_dataset(n=20000, out_path=None, seed=42):
    """
    Generates a dataset of 1-digit addition problems.

    Args:
        n (int): number of samples
        out_path (str): path to save JSONL file
        seed (int): random seed for reproducibility
    """
    random.seed(seed)

    if out_path is None:
        raise ValueError("out_path must be provided")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        for _ in range(n):
            sample = generate_example()
            f.write(json.dumps(sample) + "\n")

    print(f"Generated Add1Digit dataset: {n} samples")
    print(f"Saved to: {out_path}")


# --------------------------------------------------
# Entry point
# --------------------------------------------------
if __name__ == "__main__":
    generate_dataset(
        n=20000,
        out_path="data/synthetic/arithmetic/add_1digit.jsonl",
    )