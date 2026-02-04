from entities.base import EntityBase
from state import GlobalState


class ResultSignResolverEntity(EntityBase):
    """
    Resolves sign and routing for arithmetic results.

    Purpose:
    - Decide whether operation is ADD or SUB
    - Decide final sign
    - Decide operand ordering (|A| >= |B|)
    - Prevent illegal mixed-sign execution

    This entity:
    - NEVER computes digits
    - NEVER calls arithmetic controllers
    - ONLY prepares state for execution
    """

    name = "ResultSignResolverEntity"

    requires = [
        "arithmetic.digits_a",
        "arithmetic.digits_b",
        "arithmetic.sign",          # (sign_a, sign_b)
        "arithmetic.operation",    # "add" | "sub"
    ]

    produces = [
        "arithmetic.route",        # "add" | "sub"
        "arithmetic.sign",
        "arithmetic.digits_a",
        "arithmetic.digits_b",
    ]

    invariants = [
        "arithmetic.sign",
        "arithmetic.digits_a",
        "arithmetic.digits_b",
    ]

    # --------------------------------------------------
    # Execution
    # --------------------------------------------------
    def forward(self, state: GlobalState) -> None:

        A = state.arithmetic.digits_a
        B = state.arithmetic.digits_b
        sign_a, sign_b = state.arithmetic.sign
        op = state.arithmetic.operation

        # --------------------------------------------------
        # Zero shortcut (canonical +0)
        # --------------------------------------------------
        if A == [0] and B == [0]:
            state.arithmetic.route = "add"
            state.arithmetic.sign = 1
            state.trace.decisions.append("0 ⊕ 0 → +0")
            return

        # --------------------------------------------------
        # ADDITION
        # --------------------------------------------------
        if op == "add":
            if sign_a == sign_b:
                state.arithmetic.route = "add"
                state.arithmetic.sign = sign_a
                state.trace.decisions.append("Same-sign addition")
                return

            # Mixed signs → subtraction
            state.arithmetic.route = "sub"

            if self._abs_ge(A, B):
                state.arithmetic.sign = sign_a
                state.trace.decisions.append("|A| ≥ |B| → A - B")
            else:
                state.arithmetic.digits_a = B
                state.arithmetic.digits_b = A
                state.arithmetic.sign = sign_b
                state.trace.decisions.append("|B| > |A| → B - A")

            return

        # --------------------------------------------------
        # SUBTRACTION
        # --------------------------------------------------
        if op == "sub":
            if sign_a != sign_b:
                # a - (-b) or (-a) - b → addition
                state.arithmetic.route = "add"
                state.arithmetic.sign = sign_a
                state.trace.decisions.append(
                    "Mixed-sign subtraction → addition"
                )
                return

            # Same-sign subtraction
            state.arithmetic.route = "sub"

            if self._abs_ge(A, B):
                state.arithmetic.sign = sign_a
                state.trace.decisions.append("|A| ≥ |B| → A - B")
            else:
                state.arithmetic.digits_a = B
                state.arithmetic.digits_b = A
                state.arithmetic.sign = -sign_a
                state.trace.decisions.append("|B| > |A| → -(B - A)")

            return

        # --------------------------------------------------
        # Illegal
        # --------------------------------------------------
        raise RuntimeError(f"Unknown operation: {op}")

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    @staticmethod
    def _abs_ge(A, B):
        """
        Returns True if |A| >= |B| (LSB-first).
        """
        if len(A) != len(B):
            return len(A) > len(B)
        return A[::-1] >= B[::-1]