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

# ADD
from entities.arithmetic.add.add_1digit import Add1DigitEntity
from entities.arithmetic.add.carry import CarryEntity
from entities.arithmetic.add.add_ndigit import AddNDigitEntity

# MULTIPLY
from entities.arithmetic.multiply.multiply_1digit import Multiply1DigitEntity
from entities.arithmetic.multiply.mul_carry import MulCarryEntity
from entities.arithmetic.multiply.multiply_ndigit import MultiplyNDigitController

# SUBTRACT
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


def run_case(a, b, c, d, add_ctrl, mul_ctrl, sub_ctrl, verbose=False):
    """
    Computes: (a + b) * c - d
    """

    # -------------------------
    # ADDITION
    # -------------------------
    state = GlobalState()
    state.arithmetic.digits_a = int_to_digits(a)
    state.arithmetic.digits_b = int_to_digits(b)

    add_ctrl.forward(state)
    add_result = state.arithmetic.result

    # -------------------------
    # MULTIPLICATION
    # -------------------------
    state.trace.decisions.clear()
    state.arithmetic.digits_a = int_to_digits(add_result)
    state.arithmetic.digits_b = int_to_digits(c)

    mul_ctrl.forward(state)
    mul_result = int("".join(map(str, reversed(state.arithmetic.product))))

    # -------------------------
    # SUBTRACTION
    # -------------------------
    state.trace.decisions.clear()
    state.arithmetic.digits_a = int_to_digits(mul_result)
    state.arithmetic.digits_b = int_to_digits(d)

    sub_ctrl.forward(state)
    final_result = state.arithmetic.result

    expected = (a + b) * c - d

    if verbose:
        print(f"\nCASE: ({a} + {b}) * {c} - {d}")
        print(f"Expected: {expected}")
        print(f"Got     : {final_result}")
        print("Trace:")
        for t in state.trace.decisions:
            print(" ", t)

    assert final_result == expected, (
        f"FAILED: ({a}+{b})*{c}-{d} → {final_result} (expected {expected})"
    )


# --------------------------------------------------
# Test suites
# --------------------------------------------------
def test_edge_cases(add_ctrl, mul_ctrl, sub_ctrl):
    cases = [
        (0, 0, 0, 0),
        (1, 0, 1, 0),
        (9, 1, 5, 10),
        (99, 1, 9, 50),
        (123, 456, 0, 100),
        (10**10, 1, 9, 10**9),
        (999, 1, 999, 998),
    ]

    for a, b, c, d in cases:
        run_case(a, b, c, d, add_ctrl, mul_ctrl, sub_ctrl)


def test_random_cases(add_ctrl, mul_ctrl, sub_ctrl, n=300, max_digits=20):
    for _ in range(n):
        a = random.randint(0, 10**max_digits)
        b = random.randint(0, 10**max_digits)
        c = random.randint(0, 10**max_digits)

        intermediate = (a + b) * c
        d = random.randint(0, intermediate)  # keep non-negative

        run_case(a, b, c, d, add_ctrl, mul_ctrl, sub_ctrl)


# --------------------------------------------------
# Main entry
# --------------------------------------------------
def test_add_mul_sub_chain():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------
    # ADD entities
    # -------------------------
    add1 = Add1DigitEntity(d_model=64).to(device)
    carry = CarryEntity(d_model=64).to(device)

    add1.load_state_dict(torch.load(
        "models/entities/arithmetic/add/add_1digit/model.pt",
        map_location=device,
    ))
    carry.load_state_dict(torch.load(
        "models/entities/arithmetic/add/carry/model.pt",
        map_location=device,
    ))

    add1.eval()
    carry.eval()

    add_ctrl = AddNDigitEntity(add1, carry)

    # -------------------------
    # MULTIPLY entities
    # -------------------------
    mul1 = Multiply1DigitEntity(d_model=64).to(device)
    mulc = MulCarryEntity(d_model=32).to(device)

    mul1.load_state_dict(torch.load(
        "models/entities/arithmetic/multiply/multiply_1digit/model.pt",
        map_location=device,
    ))
    mulc.load_state_dict(torch.load(
        "models/entities/arithmetic/multiply/mul_carry/model.pt",
        map_location=device,
    ))

    mul1.eval()
    mulc.eval()

    mul_ctrl = MultiplyNDigitController(
        mul1digit_entity=mul1,
        mulcarry_entity=mulc,
        add1digit_entity=add1,
        addcarry_entity=carry,
    )

    # -------------------------
    # SUBTRACT entities
    # -------------------------
    sub1 = Subtract1DigitEntity(d_model=64).to(device)
    borrow = BorrowEntity(d_model=32).to(device)

    sub1.load_state_dict(torch.load(
        "models/entities/arithmetic/subtract/subtract_1digit/model.pt",
        map_location=device,
    ))
    borrow.load_state_dict(torch.load(
        "models/entities/arithmetic/subtract/borrow/model.pt",
        map_location=device,
    ))

    sub1.eval()
    borrow.eval()

    sub_ctrl = SubtractNDigitEntity(sub1, borrow)

    # -------------------------
    # DEMO TRACE
    # -------------------------
    print("\n--- DEMO TRACE: (123 + 77) * 9 - 456 ---")
    run_case(
        123, 77, 9, 456,
        add_ctrl, mul_ctrl, sub_ctrl,
        verbose=True,
    )

    # -------------------------
    # VALIDATION
    # -------------------------
    test_edge_cases(add_ctrl, mul_ctrl, sub_ctrl)
    test_random_cases(add_ctrl, mul_ctrl, sub_ctrl)

    print("\n✅ All DEMA add → multiply → subtract tests passed.")


if __name__ == "__main__":
    test_add_mul_sub_chain()