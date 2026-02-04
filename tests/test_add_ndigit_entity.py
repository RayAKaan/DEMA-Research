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


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def int_to_digits(n: int):
    if n == 0:
        return [0]
    return [int(d) for d in reversed(str(n))]


def run_case(a, b, controller, verbose=False, final_report=False):
    """
    Runs a single addition case.

    verbose=True       → prints reasoning trace
    final_report=True  → prints inference statistics (ONCE)
    """
    state = GlobalState()
    state.arithmetic.digits_a = int_to_digits(a)
    state.arithmetic.digits_b = int_to_digits(b)

    # 🔑 Control inference reporting
    state.control.final_report = final_report

    controller.forward(state)

    expected = a + b

    if verbose:
        print(f"\nResult: {state.arithmetic.result}")
        print("Trace:")
        for t in state.trace.decisions:
            print(" ", t)

    assert state.arithmetic.result == expected, (
        f"FAILED: {a} + {b} → {state.arithmetic.result} "
        f"(expected {expected})"
    )


# --------------------------------------------------
# Test suites
# --------------------------------------------------
def test_edge_cases(controller):
    cases = [
        (0, 0),
        (9, 1),
        (99, 1),
        (999, 1),
        (9999, 1),
        (123, 456),
        (999, 999),
        (10**20 - 1, 1),
    ]

    for a, b in cases:
        run_case(a, b, controller, verbose=False, final_report=False)


def test_random_cases(controller, num_tests=1000, max_digits=50):
    for _ in range(num_tests):
        a = random.randint(0, 10**max_digits)
        b = random.randint(0, 10**max_digits)
        run_case(a, b, controller, verbose=False, final_report=False)


# --------------------------------------------------
# Main test entry
# --------------------------------------------------
def test_add_ndigit_entity():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------
    # Load trained entities
    # --------------------------------------------------
    add1 = Add1DigitEntity(d_model=64).to(device)
    carry = CarryEntity(d_model=64).to(device)

    add1.load_state_dict(
        torch.load(
            "models/entities/arithmetic/add_1digit/model.pt",
            map_location=device,
        )
    )
    carry.load_state_dict(
        torch.load(
            "models/entities/arithmetic/carry/model.pt",
            map_location=device,
        )
    )

    add1.eval()
    carry.eval()

    controller = AddNDigitEntity(add1, carry)

    # --------------------------------------------------
    # 🔹 FULL TRACE + INFERENCE STATS (DEMO CASE)
    # --------------------------------------------------
    print("\n--- DEMO TRACE (999 + 712) ---")
    run_case(
        999,
        712,
        controller,
        verbose=True,
        final_report=True,   # 🔥 prints inference summary ONCE
    )

    # --------------------------------------------------
    # 🔹 Silent validation suites
    # --------------------------------------------------
    test_edge_cases(controller)
    test_random_cases(controller)

    print("\n✅ All DEMA arithmetic tests passed.")


# --------------------------------------------------
# Manual execution
# --------------------------------------------------
if __name__ == "__main__":
    test_add_ndigit_entity()