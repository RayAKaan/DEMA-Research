import random
import json

def generate_example(digits=3):
    lo = 10**(digits - 1)
    hi = 10**digits - 1
    a = random.randint(lo, hi)
    b = random.randint(1, hi)
    inp = f"{a}/{b}"
    out = f"{a//b} rem {a % b}"
    return {"input": inp, "output": out}

def generate_dataset(n=30000, out_path=None, digits=3):
    data = [generate_example(digits) for _ in range(n)]

    if out_path:
        with open(out_path, "w") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")

    return data

if __name__ == "__main__":
    generate_dataset(n=30000, out_path="divide_ndigit.jsonl", digits=3)