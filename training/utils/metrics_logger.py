import csv
import os

class CSVLogger:
    def __init__(self, path, fieldnames):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file = open(path, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()

    def log(self, row: dict):
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()