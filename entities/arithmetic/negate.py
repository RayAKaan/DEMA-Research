from entities.base import EntityBase
from state import GlobalState


class NegateNDigitEntity(EntityBase):
    """
    Symbolic sign negation.

    Turns +A into -A and vice versa.
    """

    name = "NegateNDigitEntity"

    requires = ["arithmetic.sign"]
    produces = ["arithmetic.sign"]
    invariants = []

    def forward(self, state: GlobalState) -> None:
        state.arithmetic.sign *= -1

        state.trace.decisions.append(
            f"NegateNDigitEntity flipped sign to {state.arithmetic.sign}"
        )