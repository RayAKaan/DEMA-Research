import random
import json


def generate_example():
    """
    Generates subtraction supervision for:
        Subtract1DigitEntity

    Learns:
        (a, b, borrow_in) -> (a - b - borrow_in) % 10
    """
    a = random.randint(0, 9)
    b = random.randint(0, 9)

    samples = []

    for borrow_in in [0, 1]:
        diff = a - b - borrow_in
        target = diff % 10

        samples.append({
            "a": a,
            "b": b,
            "borrow_in": borrow_in,
            "target": target,
        })

    return samples


def generate_dataset(n=20000, out_path=None, seed=42):
    """
    Generates a dataset for Subtract1DigitEntity.

    Args:
        n (int): number of base (a,b) pairs
        out_path (str): output JSONL path
        seed (int): RNG seed
    """
    random.seed(seed)

    all_samples = []

    for _ in range(n):
        all_samples.extend(generate_example())

    if out_path is not None:
        with open(out_path, "w") as f:
            for sample in all_samples:
                f.write(json.dumps(sample) + "\n")

    print(f"Generated {len(all_samples)} subtraction samples")
    if out_path:
        print(f"Saved to: {out_path}")

    return all_samples


if __name__ == "__main__":
    generate_dataset(
        n=20000,
        out_path="mega-math-model/data/synthetic/arithmetic/subtract_1digit.jsonl",
    )