from entities.base import EntityBase
from state import GlobalState


class NegativeNumberHandler(EntityBase):
    """
    Symbolic Entity that normalizes subtraction when the result would be negative.

    Responsibility:
    - Detect when a < b
    - Flip subtraction into (b - a)
    - Record sign in GlobalState
    - Ensure downstream entities operate on non-negative integers only

    This entity does NOT perform arithmetic.
    It only prepares the state.
    """

    name = "NegativeNumberHandler"

    requires = [
        "arithmetic.digits_a",
        "arithmetic.digits_b",
    ]

    produces = [
        "arithmetic.is_negative",
        "arithmetic.digits_a",
        "arithmetic.digits_b",
    ]

    invariants = []

    def forward(self, state: GlobalState) -> None:
        """
        Mutates state to ensure digits_a >= digits_b.
        Sets sign flag if flip occurs.
        """

        a_digits = state.arithmetic.digits_a
        b_digits = state.arithmetic.digits_b

        a_val = int("".join(map(str, reversed(a_digits))))
        b_val = int("".join(map(str, reversed(b_digits))))

        if a_val < b_val:
            # Flip operands
            state.arithmetic.digits_a = b_digits
            state.arithmetic.digits_b = a_digits
            state.arithmetic.is_negative = True

            state.trace.decisions.append(
                f"NegativeNumberHandler: flipped operands ({a_val} < {b_val})"
            )
        else:
            state.arithmetic.is_negative = False

            state.trace.decisions.append(
                f"NegativeNumberHandler: no flip ({a_val} >= {b_val})"
            )