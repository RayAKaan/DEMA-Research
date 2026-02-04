from entities.base import EntityBase
from state import GlobalState


class SignRouterEntity(EntityBase):
    """
    Routes signed arithmetic into unsigned digit-space.

    Responsibilities:
        - Extract absolute values
        - Preserve original signs
        - Decide output sign based on operation
        - Ensure digit entities never see negatives
    """

    name = "SignRouterEntity"

    requires = [
        "arithmetic.digits_a",
        "arithmetic.digits_b",
        "arithmetic.sign_a",
        "arithmetic.sign_b",
        "control.operation",   # "add" | "sub" | "mul"
    ]

    produces = [
        "arithmetic.sign",
    ]

    invariants = [
        "arithmetic.digits_a",
        "arithmetic.digits_b",
    ]

    def forward(self, state: GlobalState) -> None:

        sa = state.arithmetic.sign_a
        sb = state.arithmetic.sign_b
        op = state.control.operation

        # --------------------------------------------------
        # Zero canonicalization
        # --------------------------------------------------
        if state.arithmetic.digits_a == [0]:
            sa = 1
        if state.arithmetic.digits_b == [0]:
            sb = 1

        # --------------------------------------------------
        # Operation-specific sign logic
        # --------------------------------------------------
        if op == "add":
            # sign resolved AFTER magnitude comparison
            state.arithmetic.pending_sign_logic = "add"

        elif op == "sub":
            # a - b  →  a + (-b)
            state.arithmetic.sign_b = -sb
            state.arithmetic.pending_sign_logic = "add"

        elif op == "mul":
            state.arithmetic.sign = sa * sb
            state.arithmetic.pending_sign_logic = None

        else:
            raise ValueError(f"Unknown operation: {op}")

        state.trace.decisions.append(
            f"SignRouterEntity: op={op}, sign_a={sa}, sign_b={sb}"
        )