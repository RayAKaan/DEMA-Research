class ArithmeticInspector:
    """
    Read-only inspector for arithmetic state.

    Enforces global invariants across:
    - sign
    - digits
    - numeric consistency

    This is NOT an Entity.
    This MUST NOT mutate state.
    """

    # --------------------------------------------------
    # Invariant checks
    # --------------------------------------------------
    @staticmethod
    def check_invariants(state):

        ar = state.arithmetic

        # --------------------------------------------------
        # Sign invariant
        # --------------------------------------------------
        if hasattr(ar, "sign"):
            if ar.sign not in (-1, 1):
                raise AssertionError(
                    f"Invalid sign: {ar.sign}"
                )

        # --------------------------------------------------
        # Digit invariants
        # --------------------------------------------------
        if hasattr(ar, "result_digits") and ar.result_digits is not None:
            digits = ar.result_digits

            if not isinstance(digits, list) or len(digits) == 0:
                raise AssertionError(
                    "result_digits must be a non-empty list"
                )

            for d in digits:
                if not isinstance(d, int) or not (0 <= d <= 9):
                    raise AssertionError(
                        f"Invalid digit: {d}"
                    )

            # No leading zeros (except canonical zero)
            if len(digits) > 1 and digits[-1] == 0:
                raise AssertionError(
                    f"Leading zero in result_digits: {digits}"
                )

            # Canonical zero form
            if digits == [0]:
                if getattr(ar, "sign", 1) != 1:
                    raise AssertionError(
                        "Zero must have sign +1"
                    )
                if getattr(ar, "result", 0) != 0:
                    raise AssertionError(
                        "Zero digits but result != 0"
                    )

        # --------------------------------------------------
        # Numeric consistency
        # --------------------------------------------------
        if (
            hasattr(ar, "result") and
            hasattr(ar, "result_digits") and
            ar.result_digits is not None
        ):
            magnitude = int(
                "".join(map(str, reversed(ar.result_digits)))
            )

            sign = getattr(ar, "sign", 1)
            expected = sign * magnitude

            if ar.result != expected:
                raise AssertionError(
                    f"Result mismatch: expected {expected}, "
                    f"got {ar.result}"
                )

    # --------------------------------------------------
    # Human-readable summary
    # --------------------------------------------------
    @staticmethod
    def summary(state):
        return {
            "sign": getattr(state.arithmetic, "sign", None),
            "digits": getattr(state.arithmetic, "result_digits", None),
            "result": getattr(state.arithmetic, "result", None),
            "entities_used": list(state.trace.entities_used),
            "steps": state.control.step,
        }

    # --------------------------------------------------
    # Trace printer
    # --------------------------------------------------
    @staticmethod
    def print_trace(state):
        print("\n--- TRACE ---")
        for t in state.trace.decisions:
            print(" ", t)