from adapters.base import InputAdapter
from state import GlobalState


class ArithmeticInputAdapter(InputAdapter):
    """
    Handles pure arithmetic inputs like:
    - 6+5
    - 432+245
    """

    priority = 10  # least specific

    def supports(self, raw_input: str) -> bool:
        return (
            "+" in raw_input
            and raw_input.replace("+", "").isdigit()
        )

    def adapt(self, raw_input: str) -> GlobalState:
        a_str, b_str = raw_input.split("+")

        state = GlobalState()

        # Control layer
        state.control.goal = "add"
        state.control.done = False
        state.control.step = 0

        # Arithmetic layer
        state.arithmetic.digits_a = [int(d) for d in a_str[::-1]]
        state.arithmetic.digits_b = [int(d) for d in b_str[::-1]]
        state.arithmetic.carry = 0
        state.arithmetic.result_digits = None
        state.arithmetic.result = None

        return state
