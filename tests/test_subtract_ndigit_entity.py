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
from entities.arithmetic.subtract.subtract_1digit import Subtract1DigitEntity
from entities.arithmetic.subtract.borrow import BorrowEntity
from entities.arithmetic.subtract.subtract_ndigit import SubtractNDigitEntity


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def int_to_digits(n: int):
    """
    Converts integer to LSB-first digit list.
    Example: 712 -> [2,1,7]
    """
    if n == 0:
        return [0]
    return [int(d) for d in reversed(str(n))]


def run_case(a, b, controller, verbose=False):
    """
    Runs a single subtraction case: a - b.

    verbose=True → prints reasoning trace
    """
    assert a >= b, "This test assumes non-negative subtraction (a >= b)"

    state = GlobalState()
    state.arithmetic.digits_a = int_to_digits(a)
    state.arithmetic.digits_b = int_to_digits(b)

    controller.forward(state)

    expected = a - b

    if verbose:
        print(f"\nResult: {state.arithmetic.result}")
        print("Trace:")
        for t in state.trace.decisions:
            print(" ", t)

    assert state.arithmetic.result == expected, (
        f"FAILED: {a} - {b} → {state.arithmetic.result} "
        f"(expected {expected})"
    )


# --------------------------------------------------
# Test suites
# --------------------------------------------------
def test_edge_cases(controller):
    cases = [
        (0, 0),
        (9, 0),
        (10, 1),
        (100, 1),
        (1000, 1),
        (1234, 234),
        (999, 998),
        (10**20, 1),
    ]

    for a, b in cases:
        run_case(a, b, controller, verbose=False)


def test_random_cases(controller, num_tests=1000, max_digits=50):
    for _ in range(num_tests):
        a = random.randint(0, 10**max_digits)
        b = random.randint(0, a)  # ensure a >= b
        run_case(a, b, controller, verbose=False)


# --------------------------------------------------
# Main test entry
# --------------------------------------------------
def test_subtract_ndigit_entity():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------
    # Load trained atomic entities
    # --------------------------------------------------
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

    controller = SubtractNDigitEntity(sub1, borrow)

    # --------------------------------------------------
    # 🔹 FULL TRACE (DEMO CASE)
    # --------------------------------------------------
    print("\n--- DEMO TRACE (1000 - 712) ---")
    run_case(
        1000,
        712,
        controller,
        verbose=True,
    )

    # --------------------------------------------------
    # 🔹 Silent validation suites
    # --------------------------------------------------
    test_edge_cases(controller)
    test_random_cases(controller)

    print("\n✅ All DEMA subtraction tests passed.")


# --------------------------------------------------
# Manual execution
# --------------------------------------------------
if __name__ == "__main__":
    test_subtract_ndigit_entity()