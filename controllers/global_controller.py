from state import GlobalState

from controllers.expression_controller import ExpressionController
from entities.arithmetic.helpers.negative_number_handler import NegativeNumberHandler


class GlobalController:
    """
    Global Controller (Dimension 3)

    Operates at the highest abstraction level in DEMA.
    It performs dimensional routing and delegates reasoning
    to lower-dimensional controllers.

    Dimension semantics:
        D3: GlobalController        (why / what)
        D2: ExpressionController    (composition)
        D1: NDigit Controllers      (algorithms)
        D0: Neural Entities         (execution)
    """

    name = "GlobalController"

    def __init__(
        self,
        expression_controller: ExpressionController,
        negative_handler: NegativeNumberHandler,
    ):
        # D2 controller
        self.expression_controller = expression_controller

        # Semantic entities (dimension-agnostic)
        self.negative_handler = negative_handler

    # --------------------------------------------------
    # Public API (single entry point)
    # --------------------------------------------------
    def solve(self, problem, state: GlobalState) -> int:
        """
        Entry point for all reasoning.

        `problem` is a symbolic representation (e.g. expression list).
        """

        state.trace.decisions.append(
            "D3(Global): start reasoning"
        )

        # --------------------------------------------------
        # 1. Domain identification (hard-coded for now)
        # --------------------------------------------------
        domain = self._identify_domain(problem)
        state.trace.decisions.append(
            f"D3(Global): domain = {domain}"
        )

        if domain != "arithmetic":
            raise NotImplementedError(
                f"Domain '{domain}' not supported yet"
            )

        # --------------------------------------------------
        # 2. Dimensional projection: Global → Expression
        # --------------------------------------------------
        state.trace.decisions.append(
            "D3(Global): projecting to D2(Expression)"
        )

        # --------------------------------------------------
        # 3. Semantic normalization (sign handling)
        # --------------------------------------------------
        self._apply_semantics(problem, state)

        # --------------------------------------------------
        # 4. Delegate execution to D2 controller
        # --------------------------------------------------
        result = self.expression_controller.evaluate(
            problem,
            state,
        )

        # --------------------------------------------------
        # 5. Post-processing (sign reconciliation)
        # --------------------------------------------------
        if getattr(state.arithmetic, "is_negative", False):
            result = -result
            state.trace.decisions.append(
                "D3(Global): applied negative sign"
            )

        # --------------------------------------------------
        # 6. Finalization
        # --------------------------------------------------
        state.trace.decisions.append(
            "D3(Global): reasoning complete"
        )

        return result

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------
    def _identify_domain(self, problem) -> str:
        """
        Identify reasoning domain.
        Currently arithmetic-only.
        """
        return "arithmetic"

    def _apply_semantics(self, problem, state: GlobalState):
        """
        Apply semantic normalization before execution.
        """

        # Only needed if subtraction is present
        if "-" in problem:
            state.trace.decisions.append(
                "D3(Global): invoking NegativeNumberHandler"
            )
            self.negative_handler.forward(state)