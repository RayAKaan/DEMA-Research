import sys
import os
import random
import torch

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

from entities.arithmetic.add.add_1digit import Add1DigitEntity
from entities.arithmetic.add.carry import CarryEntity
from entities.arithmetic.add.add_ndigit import AddNDigitEntity

from entities.arithmetic.subtract.subtract_1digit import Subtract1DigitEntity
from entities.arithmetic.subtract.borrow import BorrowEntity
from entities.arithmetic.subtract.subtract_ndigit import SubtractNDigitEntity


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def int_to_digits(n: int):
    if n == 0:
        return [0]
    return [int(d) for d in reversed(str(n))]


def run_add_then_sub(a, b, c, add_controller, sub_controller, verbose=False):

    # -------------------------
    # Addition
    # -------------------------
    state = GlobalState()
    state.arithmetic.digits_a = int_to_digits(a)
    state.arithmetic.digits_b = int_to_digits(b)

    add_controller.forward(state)
    intermediate = state.arithmetic.result

    if verbose:
        print(f"\nAfter addition: {a} + {b} = {intermediate}")
        for t in state.trace.decisions:
            print(" ", t)

    # -------------------------
    # RESET STATE (CRITICAL)
    # -------------------------
    state.trace.decisions.clear()
    state.arithmetic.borrow = 0
    state.arithmetic.carry = 0

    # -------------------------
    # Subtraction
    # -------------------------
    state.arithmetic.digits_a = int_to_digits(intermediate)
    state.arithmetic.digits_b = int_to_digits(c)

    sub_controller.forward(state)

    result = state.arithmetic.result
    expected = (a + b) - c

    if verbose:
        print(f"\nAfter subtraction: {intermediate} - {c} = {result}")
        for t in state.trace.decisions:
            print(" ", t)

    assert result == expected, (
        f"FAILED: ({a} + {b}) - {c} → {result} "
        f"(expected {expected})"
    )


# --------------------------------------------------
# Test suites
# --------------------------------------------------
def test_edge_cases(add_ctrl, sub_ctrl):
    cases = [
        (0, 0, 0),
        (9, 1, 5),
        (99, 1, 50),
        (999, 1, 712),
        (1000, 0, 1),
        (123, 456, 200),
        (10**20 - 1, 1, 10**10),
    ]

    for a, b, c in cases:
        run_add_then_sub(a, b, c, add_ctrl, sub_ctrl, verbose=False)


def test_random_cases(add_ctrl, sub_ctrl, num_tests=500, max_digits=30):
    for _ in range(num_tests):
        a = random.randint(0, 10**max_digits)
        b = random.randint(0, 10**max_digits)
        intermediate = a + b
        c = random.randint(0, intermediate)  # keep non-negative
        run_add_then_sub(a, b, c, add_ctrl, sub_ctrl, verbose=False)


# --------------------------------------------------
# Main test entry
# --------------------------------------------------
def test_add_sub_chain():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------
    # Load addition entities
    # -------------------------
    add1 = Add1DigitEntity(d_model=64).to(device)
    carry = CarryEntity(d_model=64).to(device)

    add1.load_state_dict(
        torch.load(
            "models/entities/arithmetic/add/add_1digit/model.pt",
            map_location=device,
        )
    )
    carry.load_state_dict(
        torch.load(
            "models/entities/arithmetic/add/carry/model.pt",
            map_location=device,
        )
    )

    add1.eval()
    carry.eval()

    add_controller = AddNDigitEntity(add1, carry)

    # -------------------------
    # Load subtraction entities
    # -------------------------
    sub1 = Subtract1DigitEntity(d_model=64).to(device)
    borrow = BorrowEntity(d_model=32).to(device)

    sub1.load_state_dict(
        torch.load(
            "models/entities/arithmetic/subtract/subtract_1digit/model.pt",
            map_location=device,
        )
    )
    borrow.load_state_dict(
        torch.load(
            "models/entities/arithmetic/subtract/borrow/model.pt",
            map_location=device,
        )
    )

    sub1.eval()
    borrow.eval()

    sub_controller = SubtractNDigitEntity(sub1, borrow)

    # --------------------------------------------------
    # 🔹 DEMO TRACE
    # --------------------------------------------------
    print("\n--- DEMO TRACE: (999 + 1) - 712 ---")
    run_add_then_sub(
        939,
        5,
        512,
        add_controller,
        sub_controller,
        verbose=True,
    )

    # --------------------------------------------------
    # 🔹 Silent validation suites
    # --------------------------------------------------
    test_edge_cases(add_controller, sub_controller)
    test_random_cases(add_controller, sub_controller)

    print("\n✅ All DEMA add→subtract composition tests passed.")


# --------------------------------------------------
# Manual execution
# --------------------------------------------------
if __name__ == "__main__":
    test_add_sub_chain()