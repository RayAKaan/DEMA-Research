import json

def write_jsonl(path, data):
    with open(path, 'w') as f:
        for d in data:
            f.write(json.dumps(d) + '\\n')

def append_jsonl(path, data):
    with open(path, 'a') as f:
        for d in data:
            f.write(json.dumps(d) + '\\n')
