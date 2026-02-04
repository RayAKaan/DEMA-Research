import random
import json

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def generate_example(max_val=200):
    a = random.randint(1, max_val)
    b = random.randint(1, max_val)
    return {
        "input": f"gcd({a},{b})",
        "output": str(gcd(a, b))
    }

def generate_dataset(n=20000, out_path=None, max_val=200):
    data = [generate_example(max_val) for _ in range(n)]

    if out_path:
        with open(out_path, "w") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")

    return data

if __name__ == "__main__":
    generate_dataset(n=20000, out_path="gcd.jsonl", max_val=200)
    