from adapters.base import InputAdapter
from state import GlobalState


class CalculusInputAdapter(InputAdapter):
    """
    Handles calculus inputs like:
    - d/dx(x^2)
    """

    priority = 100  # most specific

    def supports(self, raw_input: str) -> bool:
        return "d/dx" in raw_input

    def adapt(self, raw_input: str) -> GlobalState:
        function = raw_input.replace("d/dx", "").strip()

        state = GlobalState()

        # Control layer
        state.control.goal = "differentiate"
        state.control.done = False
        state.control.step = 0

        # Calculus layer
        state.calculus.function = function
        state.calculus.derivative = None

        return state
