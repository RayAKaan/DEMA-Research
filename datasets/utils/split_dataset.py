import json
import random

def split_jsonl(path, train_ratio=0.8, val_ratio=0.1, seed=42):
    random.seed(seed)
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    random.shuffle(data)

    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = data[:n_train]
    val = data[n_train:n_train+n_val]
    test = data[n_train+n_val:]

    return train, val, test

def save_splits(train, val, test, prefix):
    import json
    with open(prefix + "_train.jsonl", 'w') as f:
        for d in train: f.write(json.dumps(d) + "\\n")
    with open(prefix + "_val.jsonl", 'w') as f:
        for d in val: f.write(json.dumps(d) + "\\n")
    with open(prefix + "_test.jsonl", 'w') as f:
        for d in test: f.write(json.dumps(d) + "\\n")

if __name__ == '__main__':
    train, val, test = split_jsonl("sample.jsonl")
    save_splits(train, val, test, "sample")
