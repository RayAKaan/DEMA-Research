from typing import List, Union

from state import GlobalState
from entities.arithmetic.add.add_ndigit import AddNDigitEntity
from entities.arithmetic.subtract.subtract_ndigit import SubtractNDigitEntity


class ExpressionController:
    """
    Symbolic controller that evaluates arithmetic expressions
    left-to-right with no brackets.

    Supported operators:
        +  (addition)
        -  (subtraction)

    This controller does NOT train.
    It composes existing NDigit entities.
    """

    name = "ExpressionController"

    def __init__(
        self,
        add_controller: AddNDigitEntity,
        subtract_controller: SubtractNDigitEntity,
    ):
        self.add = add_controller
        self.sub = subtract_controller

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def evaluate(
        self,
        expression: List[Union[int, str]],
        state: GlobalState,
    ) -> int:
        """
        Evaluate expression left-to-right.

        Example:
            [939, "+", 5, "-", 512]
        """

        assert len(expression) >= 1
        assert isinstance(expression[0], int)

        # Initialize accumulator
        acc = expression[0]

        state.trace.decisions.append(
            f"Expression start → acc = {acc}"
        )

        i = 1
        while i < len(expression):
            op = expression[i]
            rhs = expression[i + 1]

            assert op in {"+", "-"}
            assert isinstance(rhs, int)

            if op == "+":
                acc = self._run_add(acc, rhs, state)
            else:
                acc = self._run_sub(acc, rhs, state)

            i += 2

        state.trace.decisions.append(
            f"Expression result → {acc}"
        )

        return acc

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------
    def _run_add(self, a: int, b: int, state: GlobalState) -> int:
        state.trace.decisions.append(
            f"Apply ADD → {a} + {b}"
        )

        state.arithmetic.digits_a = self._int_to_digits(a)
        state.arithmetic.digits_b = self._int_to_digits(b)

        self.add.forward(state)

        return state.arithmetic.result

    def _run_sub(self, a: int, b: int, state: GlobalState) -> int:
        state.trace.decisions.append(
            f"Apply SUB → {a} - {b}"
        )

        state.arithmetic.digits_a = self._int_to_digits(a)
        state.arithmetic.digits_b = self._int_to_digits(b)

        self.sub.forward(state)

        return state.arithmetic.result

    @staticmethod
    def _int_to_digits(n: int):
        if n == 0:
            return [0]
        return [int(d) for d in reversed(str(abs(n)))]