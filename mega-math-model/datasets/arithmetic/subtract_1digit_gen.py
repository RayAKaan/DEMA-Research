import random
import json

def generate_example():
    a = random.randint(0,9)
    b = random.randint(0,9)
    inp = f"{a}-{b}"
    out = str(a - b)
    return {"input": inp, "output": out}

def generate_dataset(n=20000, out_path=None):
    data = [generate_example() for _ in range(n)]
    if out_path:
        with open(out_path, "w") as f:
            for d in data: f.write(json.dumps(d) + "\n")
    return data

if __name__ == "__main__":
    generate_dataset(20000, "subtract_1digit.jsonl")
