import os
import sys
import importlib

# ------------------------------------------------
# FIX 1: Ensure project root is on PYTHONPATH
# ------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# ------------------------------------------------
# DATASET TARGET LIST
# ------------------------------------------------
TARGETS = [
    # --------------------------
    # ARITHMETIC (original)
    # --------------------------
    ("datasets.arithmetic.add_1digit_gen", "generate_dataset",
     "data/synthetic/arithmetic/add_1digit.jsonl", 20000),

    ("datasets.arithmetic.add_ndigit_gen", "generate_dataset",
     "data/synthetic/arithmetic/add_ndigit.jsonl", 50000),

    ("datasets.arithmetic.multiply_1digit_gen", "generate_dataset",
     "data/synthetic/arithmetic/multiply_1digit.jsonl", 20000),

    ("datasets.arithmetic.multiply_ndigit_gen", "generate_dataset",
     "data/synthetic/arithmetic/multiply_ndigit.jsonl", 30000),

    ("datasets.arithmetic.gcd_gen", "generate_dataset",
     "data/synthetic/arithmetic/gcd.jsonl", 20000),

    # --------------------------
    # ARITHMETIC (NEW HRMs)
    # --------------------------
    ("datasets.arithmetic.subtract_1digit_gen", "generate_dataset",
     "data/synthetic/arithmetic/subtract_1digit.jsonl", 20000),

    ("datasets.arithmetic.subtract_ndigit_gen", "generate_dataset",
     "data/synthetic/arithmetic/subtract_ndigit.jsonl", 50000),

    ("datasets.arithmetic.divide_1digit_gen", "generate_dataset",
     "data/synthetic/arithmetic/divide_1digit.jsonl", 20000),

    ("datasets.arithmetic.divide_ndigit_gen", "generate_dataset",
     "data/synthetic/arithmetic/divide_ndigit.jsonl", 30000),

    ("datasets.arithmetic.negative_number_handler_gen", "generate_dataset",
     "data/synthetic/arithmetic/negative_number.jsonl", 20000),

    ("datasets.arithmetic.modulo_small_gen", "generate_dataset",
     "data/synthetic/arithmetic/modulo_small.jsonl", 20000),

    # --------------------------
    # ALGEBRA
    # --------------------------
    ("datasets.algebra.simplify_gen", "generate_dataset",
     "data/synthetic/algebra/simplify.jsonl", 50000),

    ("datasets.algebra.expand_gen", "generate_dataset",
     "data/synthetic/algebra/expand.jsonl", 30000),

    ("datasets.algebra.factor_gen", "generate_dataset",
     "data/synthetic/algebra/factor.jsonl", 20000),

    ("datasets.algebra.collect_gen", "generate_dataset",
     "data/synthetic/algebra/collect.jsonl", 20000),

    ("datasets.algebra.solve_linear_gen", "generate_dataset",
     "data/synthetic/algebra/solve_linear.jsonl", 20000),

    # --------------------------
    # CALCULUS
    # --------------------------
    ("datasets.calculus.derivative_power_gen", "generate_dataset",
     "data/synthetic/calculus/derivative_power.jsonl", 30000),

    ("datasets.calculus.derivative_product_gen", "generate_dataset",
     "data/synthetic/calculus/derivative_product.jsonl", 20000),
]

# ------------------------------------------------
# MAIN GENERATION FUNCTION
# ------------------------------------------------
def main():
    os.makedirs("data/synthetic/arithmetic", exist_ok=True)
    os.makedirs("data/synthetic/algebra", exist_ok=True)
    os.makedirs("data/synthetic/calculus", exist_ok=True)

    for module_path, func_name, out_file, count in TARGETS:
        print(f"[Generating] {module_path} → {out_file}")
        mod = importlib.import_module(module_path)  # NOTE: now uses dot-notation directly
        fn = getattr(mod, func_name)
        fn(count, out_file)

    print("\nAll datasets generated successfully.")

# ------------------------------------------------
# ENTRY POINT
# ------------------------------------------------
if __name__ == "__main__":
    main()
