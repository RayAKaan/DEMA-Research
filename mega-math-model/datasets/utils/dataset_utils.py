import json
import random

def load_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data, out_path):
    with open(out_path, 'w') as f:
        for d in data:
            f.write(json.dumps(d) + '\\n')

def shuffle_data(data, seed=42):
    random.seed(seed)
    random.shuffle(data)
    return data

def preview(data, n=5):
    for i in range(min(n, len(data))):
        print(data[i])
