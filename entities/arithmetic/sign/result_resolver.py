from entities.base import EntityBase
from state import GlobalState


class ResultSignResolverEntity(EntityBase):
    """
    Resolves final sign after add/sub based on magnitudes.

    Requires:
        - abs(A), abs(B)
        - intermediate unsigned result
        - pending sign logic
    """

    name = "ResultSignResolverEntity"

    requires = [
        "arithmetic.result",
        "arithmetic.sign_a",
        "arithmetic.sign_b",
        "arithmetic.pending_sign_logic",
    ]

    produces = [
        "arithmetic.sign",
    ]

    invariants = [
        "arithmetic.result",
    ]

    def forward(self, state: GlobalState) -> None:

        logic = state.arithmetic.pending_sign_logic
        sa = state.arithmetic.sign_a
        sb = state.arithmetic.sign_b

        if logic != "add":
            return

        # Compare absolute magnitudes
        a = abs(state.arithmetic.a)
        b = abs(state.arithmetic.b)

        if a > b:
            state.arithmetic.sign = sa
        elif b > a:
            state.arithmetic.sign = sb
        else:
            # Result is zero
            state.arithmetic.sign = 1

        state.arithmetic.pending_sign_logic = None

        state.trace.decisions.append(
            f"ResultSignResolverEntity: final_sign={state.arithmetic.sign}"
        )