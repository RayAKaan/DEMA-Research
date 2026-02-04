from entities.base import EntityBase
from state import GlobalState


class NegateNDigitEntity(EntityBase):
    """
    Symbolic entity that negates a number.

    Operation:
        x → -x

    Rules:
        - digits are unchanged
        - sign is flipped
        - zero is always +0
    """

    name = "NegateNDigitEntity"

    requires = [
        "arithmetic.digits",
        "arithmetic.sign",
    ]

    produces = [
        "arithmetic.sign",
    ]

    invariants = [
        "arithmetic.digits",
    ]

    def forward(self, state: GlobalState) -> None:

        digits = state.arithmetic.digits
        sign = state.arithmetic.sign

        # Zero canonicalization
        if digits == [0]:
            state.arithmetic.sign = 1
            state.trace.decisions.append("Negate: zero → +0")
            return

        # Flip sign
        state.arithmetic.sign = -sign

        state.trace.decisions.append(
            f"NegateNDigitEntity: sign {sign} → {-sign}"
        )