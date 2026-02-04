import sys
import os
import random
import torch
from collections import defaultdict
from time import time

# --------------------------------------------------
# Path setup
# --------------------------------------------------
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# --------------------------------------------------
# Imports
# --------------------------------------------------
from state import GlobalState

from entities.arithmetic.multiply.multiply_1digit import Multiply1DigitEntity
from entities.arithmetic.multiply.mul_carry import MulCarryEntity
from entities.arithmetic.add.add_1digit import Add1DigitEntity
from entities.arithmetic.add.carry import CarryEntity
from entities.arithmetic.multiply.multiply_ndigit import MultiplyNDigitController


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def int_to_digits(n: int):
    if n == 0:
        return [0]
    return [int(d) for d in reversed(str(n))]


def digits_to_int(digits):
    return int("".join(map(str, reversed(digits))))


def run_case(a, b, controller, verbose_on_fail=False):
    state = GlobalState()
    state.arithmetic.digits_a = int_to_digits(a)
    state.arithmetic.digits_b = int_to_digits(b)

    controller.forward(state)

    result = digits_to_int(state.arithmetic.product)
    expected = a * b
    ok = (result == expected)

    if not ok and verbose_on_fail:
        print("\n❌ FAILURE TRACE")
        print(f"Case     : {a} × {b}")
        print(f"Expected : {expected}")
        print(f"Got      : {result}")
        print("Trace:")
        for t in state.trace.decisions:
            print(" ", t)

    return ok, result, expected


# --------------------------------------------------
# Test suites
# --------------------------------------------------
def run_edge_cases(controller, report):
    cases = [
        (0, 0),
        (9, 9),
        (10, 10),
        (99, 99),
        (999, 999),
        (100, 100),
        (10**10, 9),
        (10**20, 1),
    ]

    print("\n[EDGE CASES]")
    for a, b in cases:
        ok, _, _ = run_case(a, b, controller, verbose_on_fail=True)
        report["edge"]["total"] += 1
        if not ok:
            report["edge"]["fail"] += 1


def run_carry_stress(controller, report):
    print("\n[CARRY STRESS]")
    for k in range(1, 9):
        a = int("9" * k)
        b = int("9" * k)

        print(f"  Carry chain length {k}")
        ok, _, _ = run_case(a, b, controller, verbose_on_fail=True)
        report["carry_chain"]["total"] += 1
        if not ok:
            report["carry_chain"]["fail"] += 1


def run_random_cases(
    controller,
    report,
    n=100,
    max_digits=20,
    progress_every=10,
):
    print(f"\n[RANDOM TESTS] n={n}, max_digits={max_digits}")
    start = time()

    for i in range(1, n + 1):
        if i % progress_every == 0 or i == 1:
            elapsed = time() - start
            print(
                f"  [{i:4d}/{n}] "
                f"elapsed={elapsed:.1f}s"
            )

        a = random.randint(0, 10**max_digits)
        b = random.randint(0, 10**max_digits)

        ok, _, _ = run_case(a, b, controller)
        report["random"]["total"] += 1
        if not ok:
            report["random"]["fail"] += 1
            # Uncomment for fail-fast:
            # run_case(a, b, controller, verbose_on_fail=True)
            # raise RuntimeError("Random test failure")


# --------------------------------------------------
# Main test entry
# --------------------------------------------------
def test_multiply_ndigit_entity():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------
    # Load trained entities
    # -------------------------------
    mul1 = Multiply1DigitEntity(d_model=64).to(device)
    mulc = MulCarryEntity(d_model=32).to(device)
    add1 = Add1DigitEntity(d_model=64).to(device)
    addc = CarryEntity(d_model=64).to(device)

    mul1.load_state_dict(torch.load(
        "models/entities/arithmetic/multiply/multiply_1digit/model.pt",
        map_location=device,
    ))
    mulc.load_state_dict(torch.load(
        "models/entities/arithmetic/multiply/mul_carry/model.pt",
        map_location=device,
    ))
    add1.load_state_dict(torch.load(
        "models/entities/arithmetic/add/add_1digit/model.pt",
        map_location=device,
    ))
    addc.load_state_dict(torch.load(
        "models/entities/arithmetic/add/carry/model.pt",
        map_location=device,
    ))

    mul1.eval()
    mulc.eval()
    add1.eval()
    addc.eval()

    controller = MultiplyNDigitController(
        mul1digit_entity=mul1,
        mulcarry_entity=mulc,
        add1digit_entity=add1,
        addcarry_entity=addc,
    )

    # -------------------------------
    # Demo sanity check
    # -------------------------------
    print("\n--- DEMO TRACE (123 × 456) ---")
    run_case(123, 456, controller, verbose_on_fail=True)

    # -------------------------------
    # Structured evaluation
    # -------------------------------
    report = defaultdict(lambda: {"total": 0, "fail": 0})

    run_edge_cases(controller, report)
    run_carry_stress(controller, report)
    run_random_cases(
        controller,
        report,
        n=100,          # FAST default
        max_digits=20,  # SAFE default
        progress_every=10,
    )

    # -------------------------------
    # Summary
    # -------------------------------
    print("\n========== TEST SUMMARY ==========")
    for k, v in report.items():
        total = v["total"]
        fail = v["fail"]
        acc = 1 - fail / max(1, total)
        print(
            f"{k.upper():>12} : "
            f"{total:4d} tests | "
            f"{fail:3d} fail | "
            f"accuracy = {acc:.3f}"
        )

    if any(v["fail"] > 0 for v in report.values()):
        print("\n❌ Multiplication is NOT fully correct yet.")
    else:
        print("\n✅ All multiplication tests passed.")


# --------------------------------------------------
# Entry point
# --------------------------------------------------
if __name__ == "__main__":
    test_multiply_ndigit_entity()