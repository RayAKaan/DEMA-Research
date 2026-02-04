import random
import json

def generate_example(max_val=200):
    a = random.randint(0, max_val)
    b = random.randint(1, max_val)
    return {"input": f"{a}%{b}", "output": str(a % b)}

def generate_dataset(n=20000, out_path=None, max_val=200):
    data = [generate_example(max_val) for _ in range(n)]

    if out_path:
        with open(out_path, "w") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")

    return data

if __name__ == "__main__":
    generate_dataset(n=20000, out_path="modulo_small.jsonl", max_val=200)