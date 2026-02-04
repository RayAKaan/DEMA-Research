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

from controllers.expression_controller import ExpressionController


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def eval_python(expr):
    """Reference evaluator (left-to-right, no precedence)."""
    acc = expr[0]
    i = 1
    while i < len(expr):
        op = expr[i]
        rhs = expr[i + 1]
        if op == "+":
            acc = acc + rhs
        else:
            acc = acc - rhs
        i += 2
    return acc


# --------------------------------------------------
# Main test
# --------------------------------------------------
def test_expression_controller():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------
    # Load trained entities
    # --------------------------------------------------
    add1 = Add1DigitEntity(d_model=64).to(device)
    carry = CarryEntity(d_model=64).to(device)

    sub1 = Subtract1DigitEntity(d_model=64).to(device)
    borrow = BorrowEntity(d_model=32).to(device)

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

    add1.eval()
    carry.eval()
    sub1.eval()
    borrow.eval()

    # --------------------------------------------------
    # Build controllers
    # --------------------------------------------------
    add_nd = AddNDigitEntity(add1, carry)
    sub_nd = SubtractNDigitEntity(sub1, borrow)

    expr_ctrl = ExpressionController(add_nd, sub_nd)

    # --------------------------------------------------
    # DEMO TRACE
    # --------------------------------------------------
    state = GlobalState()

    expr = [939, "+", 5, "-", 512]

    print("\n--- DEMO TRACE (Expression Evaluation) ---")
    result = expr_ctrl.evaluate(expr, state)

    print(f"\nExpression: {expr}")
    print(f"Result: {result}")
    print("Trace:")
    for t in state.trace.decisions:
        print(" ", t)

    expected = eval_python(expr)
    assert result == expected

    # --------------------------------------------------
    # Edge cases
    # --------------------------------------------------
    cases = [
        [0],
        [5, "+", 0],
        [10, "-", 10],
        [999, "+", 1, "-", 1000],
        [1000, "-", 1, "+", 1],
        [12345, "-", 12345],
    ]

    for expr in cases:
        state = GlobalState()
        out = expr_ctrl.evaluate(expr, state)
        assert out == eval_python(expr)

    # --------------------------------------------------
    # Randomized stress tests
    # --------------------------------------------------
    for _ in range(500):
        length = random.randint(1, 7)  # odd length
        expr = [random.randint(0, 10**6)]
        for _ in range((length - 1) // 2):
            expr.append(random.choice(["+", "-"]))
            expr.append(random.randint(0, 10**6))

        state = GlobalState()
        out = expr_ctrl.evaluate(expr, state)
        assert out == eval_python(expr), f"FAILED: {expr}"

    print("\n✅ All DEMA expression controller tests passed.")


# --------------------------------------------------
# Manual execution
# --------------------------------------------------
if __name__ == "__main__":
    test_expression_controller()