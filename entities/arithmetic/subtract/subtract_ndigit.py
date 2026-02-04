from entities.base import EntityBase
from state import GlobalState

from entities.arithmetic.subtract.subtract_1digit import Subtract1DigitEntity
from entities.arithmetic.subtract.borrow import BorrowEntity
from controllers.inspector import ArithmeticInspector


class SubtractNDigitEntity(EntityBase):
    """
    Signed N-digit subtraction controller.

    Computes:
        A - B

    ARCHITECTURE RULES:
    - Digits are ALWAYS magnitude-only
    - Borrow is explicit
    - Sign is resolved at controller level
    - Mixed-sign cases MUST be routed externally
    """

    name = "SubtractNDigitEntity"

    requires = [
        "arithmetic.digits_a",
        "arithmetic.digits_b",
        "arithmetic.sign",   # tuple: (sign_a, sign_b)
    ]

    produces = [
        "arithmetic.result",
        "arithmetic.result_digits",
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
        subtract1digit_entity: Subtract1DigitEntity,
        borrow_entity: BorrowEntity,
    ):
        super().__init__()
        self.sub1 = subtract1digit_entity
        self.borrow = borrow_entity

    # --------------------------------------------------
    # Execution
    # --------------------------------------------------
    def forward(self, state: GlobalState) -> None:

        A = state.arithmetic.digits_a  # |A|, LSB-first
        B = state.arithmetic.digits_b  # |B|, LSB-first
        sign_a, sign_b = state.arithmetic.sign

        # --------------------------------------------------
        # SAFETY: forbid mixed-sign usage here
        # --------------------------------------------------
        if sign_a != sign_b:
            raise RuntimeError(
                "SubtractNDigitEntity received mixed signs. "
                "Route via SignRouterEntity."
            )

        # --------------------------------------------------
        # Magnitude routing (ABS compare)
        # --------------------------------------------------
        if len(A) < len(B) or (len(A) == len(B) and A[::-1] < B[::-1]):
            A, B = B, A
            result_sign = -sign_a
            state.trace.decisions.append(
                "Operands swapped for magnitude comparison"
            )
        else:
            result_sign = sign_a

        max_len = max(len(A), len(B))

        borrow = 0
        result_digits = []

        # --------------------------------------------------
        # Digit-wise subtraction
        # --------------------------------------------------
        for i in range(max_len):
            a = A[i] if i < len(A) else 0
            b = B[i] if i < len(B) else 0

            # -------------------------
            # Digit computation
            # -------------------------
            state.arithmetic.digits_a = [a]
            state.arithmetic.digits_b = [b]
            state.arithmetic.borrow = borrow

            self.sub1.forward(state)
            diff = state.arithmetic.diff

            # -------------------------
            # Borrow computation
            # -------------------------
            self.borrow.forward(state)
            borrow_out = state.arithmetic.borrow

            result_digits.append(diff)

            state.trace.decisions.append(
                f"Digit {i}: a={a}, b={b}, borrow_in={borrow} "
                f"→ diff={diff}, borrow_out={borrow_out}"
            )

            borrow = borrow_out

        # --------------------------------------------------
        # Canonicalize zero (NO -0 allowed)
        # --------------------------------------------------
        magnitude = int("".join(map(str, reversed(result_digits))))

        if magnitude == 0:
            result_sign = 1
            result_digits = [0]
            state.trace.decisions.append("Canonicalized -0 to +0")

        # --------------------------------------------------
        # Final output
        # --------------------------------------------------
        state.arithmetic.result_digits = result_digits
        state.arithmetic.result = magnitude
        state.arithmetic.sign = result_sign

        state.control.done = True

        # --------------------------------------------------
        # 🔒 INVARIANT ENFORCEMENT (MANDATORY)
        # --------------------------------------------------
        ArithmeticInspector.check_invariants(state)