import re
from adapters.base import InputAdapter
from state import GlobalState


class AlgebraInputAdapter(InputAdapter):
    """
    Handles symbolic algebra inputs such as:
    - 5x + 7y = 0
    - (x + x)^2
    - x^2 + 3x + 2

    Explicitly rejects function-style inputs like:
    - sin(x)
    - cos(x)
    - log(x)
    """

    priority = 50  # More specific than arithmetic, less than calculus

    def supports(self, raw_input: str) -> bool:
        raw_input = raw_input.strip()

        # Reject function-style calls like sin(x), cos(x), log(x)
        # Matches: letters immediately followed by '('
        if re.search(r"[a-zA-Z]+\s*\(", raw_input):
            return False

        # Accept algebraic expressions with variables or equations
        return any(sym in raw_input for sym in ["x", "y", "="])

    def adapt(self, raw_input: str) -> GlobalState:
        state = GlobalState()

        # Control layer
        state.control.goal = "solve_linear" if "=" in raw_input else "simplify"
        state.control.done = False
        state.control.step = 0

        # Algebra layer
        state.algebra.expr = raw_input
        state.algebra.simplified = None
        state.algebra.expanded = None
        state.algebra.factored = None
        state.algebra.solution = None

        return state
