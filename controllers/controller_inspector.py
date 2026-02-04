class ControllerInspector:
    """
    Controller-level invariant enforcer.

    This inspector is called AFTER every arithmetic chain
    (add / sub / mul / negate).

    Purpose:
    - Enforce that ResultSignResolverEntity was used
    - Prevent illegal controller usage
    - Canonicalize zero
    - Detect silent architectural violations

    This file is LOCKED.
    Do not modify casually.
    """

    # --------------------------------------------------
    # Entry point
    # --------------------------------------------------
    @staticmethod
    def inspect(state):
        ControllerInspector._check_routing(state)
        ControllerInspector._check_sign(state)
        ControllerInspector._check_digits(state)
        ControllerInspector._canonicalize_zero(state)

    # --------------------------------------------------
    # Routing invariants
    # --------------------------------------------------
    @staticmethod
    def _check_routing(state):
        """
        Ensures that arithmetic routing was resolved BEFORE execution.
        """

        route = getattr(state.arithmetic, "route", None)

        if route is None:
            raise AssertionError(
                "Arithmetic route missing.\n"
                "ResultSignResolverEntity must be called before execution."
            )

        if route not in ("add", "sub", "mul", "negate"):
            raise AssertionError(
                f"Invalid arithmetic route: {route}"
            )

    # --------------------------------------------------
    # Sign invariants
    # --------------------------------------------------
    @staticmethod
    def _check_sign(state):
        """
        Ensures sign is valid and well-formed.
        """

        sign = getattr(state.arithmetic, "sign", None)

        if sign is None:
            raise AssertionError("Arithmetic sign missing")

        # Addition / subtraction use scalar sign
        if isinstance(sign, int):
            if sign not in (-1, 1):
                raise AssertionError(f"Invalid scalar sign: {sign}")
            return

        # Multiplication uses tuple sign
        if isinstance(sign, tuple):
            if len(sign) != 2:
                raise AssertionError(f"Invalid sign tuple: {sign}")
            if not all(s in (-1, 1) for s in sign):
                raise AssertionError(f"Invalid sign tuple values: {sign}")
            return

        raise AssertionError(f"Unrecognized sign format: {sign}")

    # --------------------------------------------------
    # Digit invariants
    # --------------------------------------------------
    @staticmethod
    def _check_digits(state):
        """
        Ensures all digits are valid magnitude digits.
        """

        digits = getattr(state.arithmetic, "result_digits", None)

        if digits is None:
            return  # some controllers produce only product / result

        if not isinstance(digits, list):
            raise AssertionError("result_digits must be a list")

        for d in digits:
            if not isinstance(d, int):
                raise AssertionError(f"Non-integer digit: {d}")
            if not (0 <= d <= 9):
                raise AssertionError(f"Invalid digit value: {d}")

    # --------------------------------------------------
    # Zero canonicalization
    # --------------------------------------------------
    @staticmethod
    def _canonicalize_zero(state):
        """
        Enforces +0 canonical form.
        """

        result = getattr(state.arithmetic, "result", None)
        digits = getattr(state.arithmetic, "result_digits", None)

        if result == 0:
            state.arithmetic.sign = 1
            if digits is not None:
                state.arithmetic.result_digits = [0]