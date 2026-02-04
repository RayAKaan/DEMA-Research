import json
import random


def generate_example():
    """
    Generates carry supervision for 1-digit addition.

    Learns:
        (a, b, carry_in) -> carry_out
    """
    a = random.randint(0, 9)
    b = random.randint(0, 9)

    samples = []

    for carry_in in [0, 1]:
        carry_out = 1 if (a + b + carry_in) >= 10 else 0

        samples.append({
            "a": a,
            "b": b,
            "carry_in": carry_in,
            "target": carry_out,
        })

    return samples


def generate_dataset(n=20000, out_path=None, seed=42):
    """
    Generates a dataset for Add CarryEntity.

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

    print(f"Generated {len(all_samples)} carry samples")
    if out_path:
        print(f"Saved to: {out_path}")

    return all_samples


if __name__ == "__main__":
    generate_dataset(
        n=20000,
        out_path="data/synthetic/arithmetic/add_carry.jsonl",
    )