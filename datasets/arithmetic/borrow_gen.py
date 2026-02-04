import random
import json

def generate_example():
    a = random.randint(0, 9)
    b = random.randint(0, 9)
    borrow_in = random.randint(0, 1)

    borrow_out = 1 if (a - borrow_in) < b else 0

    return {
        "a": a,
        "b": b,
        "borrow_in": borrow_in,
        "borrow_out": borrow_out,
    }


def generate_dataset(n=20000, out_path=None):
    data = [generate_example() for _ in range(n)]

    if out_path:
        with open(out_path, "w") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")

    return data


if __name__ == "__main__":
    generate_dataset(
        n=20000,
        out_path="data/synthetic/arithmetic/borrow.jsonl"
    )