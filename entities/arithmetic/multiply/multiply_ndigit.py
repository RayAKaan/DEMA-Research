from entities.base import EntityBase
from state import GlobalState

from entities.arithmetic.add.add_ndigit import AddNDigitEntity
from controllers.inspector import ArithmeticInspector


# --------------------------------------------------
# Utility
# --------------------------------------------------
def int_to_digits(n: int):
    """
    Converts non-negative integer to LSB-first digit list.
    """
    if n == 0:
        return [0]
    return [int(d) for d in reversed(str(n))]


# --------------------------------------------------
# MultiplyNDigitController (SIGNED, MAGNITUDE-ONLY)
# --------------------------------------------------
class MultiplyNDigitController(EntityBase):
    """
    Signed symbolic N-digit multiplication controller.

    Computes:
        A * B

    Architecture rules:
    - Digit entities operate on MAGNITUDE ONLY
    - Sign is composed explicitly at controller level
    - Zero always produces canonical +0
    """

    name = "MultiplyNDigitController"

    requires = [
        "arithmetic.digits_a",
        "arithmetic.digits_b",
        "arithmetic.sign",   # tuple: (sign_a, sign_b)
    ]

    produces = [
        "arithmetic.product",
        "arithmetic.sign",
    ]

    invariants = [
        "arithmetic.digits_a",
        "arithmetic.digits_b",
        "arithmetic.sign",
    ]

    # --------------------------------------------------
    # Init
    # --------------------------------------------------
    def __init__(
        self,
        mul1digit_entity,
        mulcarry_entity,
        add1digit_entity,
        addcarry_entity,
    ):
        super().__init__()

        self.mul_digit = mul1digit_entity
        self.mul_carry = mulcarry_entity

        # Internal magnitude-only adder
        self.adder = AddNDigitEntity(
            add1digit_entity,
            addcarry_entity,
        )

    # --------------------------------------------------
    # Execution
    # --------------------------------------------------
    def forward(self, state: GlobalState) -> None:
        """
        Executes signed symbolic long multiplication.
        """

        A = state.arithmetic.digits_a  # |A|, LSB-first
        B = state.arithmetic.digits_b  # |B|, LSB-first
        sign_a, sign_b = state.arithmetic.sign

        # --------------------------------------------------
        # Zero fast-path (canonical +0)
        # --------------------------------------------------
        if A == [0] or B == [0]:
            state.arithmetic.product = [0]
            state.arithmetic.sign = 1

            state.trace.decisions.append(
                "Zero multiplication shortcut → result = +0"
            )

            state.control.done = True

            # 🔒 Inspector hook
            ArithmeticInspector.check_invariants(state)
            return

        # --------------------------------------------------
        # Compose result sign
        # --------------------------------------------------
        result_sign = sign_a * sign_b

        state.trace.decisions.append(
            f"MultiplyNDigitController: |A|={A}, |B|={B}, sign={result_sign}"
        )

        partial_rows = []

        # --------------------------------------------------
        # Build partial products (ABSOLUTE VALUES ONLY)
        # --------------------------------------------------
        for j, b_digit in enumerate(B):
            carry = 0
            row = [0] * j  # positional shift

            for a_digit in A:
                local = GlobalState()
                local.arithmetic.digits_a = [a_digit]
                local.arithmetic.digits_b = [b_digit]
                local.arithmetic.carry = carry

                # -------------------------
                # Digit prediction
                # -------------------------
                self.mul_digit.forward(local)
                prod_digit = local.arithmetic.prod

                # -------------------------
                # Carry prediction
                # -------------------------
                self.mul_carry.forward(local)
                carry = local.arithmetic.carry

                row.append(prod_digit)

            # Flush remaining carry
            if carry > 0:
                row.extend(int_to_digits(carry))

            partial_rows.append(row)

            state.trace.decisions.append(
                f"Partial row {j}: {row}"
            )

        # --------------------------------------------------
        # Sum partial rows (magnitude only)
        # --------------------------------------------------
        result = partial_rows[0]

        for i in range(1, len(partial_rows)):
            add_state = GlobalState()
            add_state.arithmetic.digits_a = result
            add_state.arithmetic.digits_b = partial_rows[i]
            add_state.arithmetic.sign = 1  # magnitude addition ONLY

            self.adder.forward(add_state)

            result = int_to_digits(add_state.arithmetic.result)

            state.trace.decisions.append(
                f"After adding row {i}: {result}"
            )

        # --------------------------------------------------
        # Final output
        # --------------------------------------------------
        state.arithmetic.product = result
        state.arithmetic.sign = result_sign

        state.trace.decisions.append(
            f"Final product digits: {result}, sign={result_sign}"
        )

        state.control.done = True

        # --------------------------------------------------
        # 🔒 Inspector hook (MANDATORY)
        # --------------------------------------------------
        ArithmeticInspector.check_invariants(state)