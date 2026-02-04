from entities.base import EntityBase
from state import GlobalState


class SignEntity(EntityBase):
    """
    Ensures arithmetic.sign exists and is valid.

    sign ∈ {+1, -1}
    """

    name = "SignEntity"

    requires = []
    produces = ["arithmetic.sign"]
    invariants = []

    def forward(self, state: GlobalState) -> None:
        # Default sign is +1
        if not hasattr(state.arithmetic, "sign"):
            state.arithmetic.sign = 1

        if state.arithmetic.sign not in (-1, 1):
            raise ValueError(
                f"Invalid sign: {state.arithmetic.sign}"
            )

        state.trace.decisions.append(
            f"SignEntity enforced sign={state.arithmetic.sign}"
        )