import time
from collections import defaultdict

from entities.base import EntityBase
from state import GlobalState

from entities.arithmetic.add.add_1digit import Add1DigitEntity
from entities.arithmetic.add.carry import CarryEntity
from controllers.inspector import ArithmeticInspector


# --------------------------------------------------
# Inference profiler (research-only, silent)
# --------------------------------------------------
class InferenceStats:
    def __init__(self):
        self.times = defaultdict(float)
        self.counts = defaultdict(int)

    def log(self, key: str, duration: float):
        self.times[key] += duration
        self.counts[key] += 1

    def report(self):
        print("\n=== Inference Summary (Final) ===")
        for k in sorted(self.times.keys()):
            total = self.times[k]
            count = self.counts[k]
            avg_ms = (total / count) * 1000 if count > 0 else 0.0
            print(
                f"{k:<28} | "
                f"calls={count:<6} | "
                f"total={total:>8.2f}s | "
                f"avg={avg_ms:>7.3f} ms"
            )


# --------------------------------------------------
# AddNDigitEntity
# --------------------------------------------------
class AddNDigitEntity(EntityBase):
    """
    Signed multi-digit addition controller (MAGNITUDE ONLY).

    RULES:
    - Operates ONLY on same-sign operands
    - Computes absolute magnitude
    - Preserves sign without applying it
    - Mixed-sign cases MUST be routed externally
    """

    name = "AddNDigitEntity"

    requires = [
        "arithmetic.digits_a",
        "arithmetic.digits_b",
        "arithmetic.sign",
    ]

    produces = [
        "arithmetic.result_digits",
        "arithmetic.result",
        "arithmetic.sign",
    ]

    invariants = [
        "arithmetic.digits_a",
        "arithmetic.digits_b",
        "arithmetic.sign",
    ]

    def __init__(
        self,
        add1digit_entity: Add1DigitEntity,
        carry_entity: CarryEntity,
    ):
        super().__init__()
        self.add1 = add1digit_entity
        self.carry = carry_entity
        self.stats = InferenceStats()

    # --------------------------------------------------
    # Execution
    # --------------------------------------------------
    def forward(self, state: GlobalState) -> None:
        start_total = time.time()

        digits_a = state.arithmetic.digits_a
        digits_b = state.arithmetic.digits_b
        sign = state.arithmetic.sign  # preserved, never applied

        # --------------------------------------------------
        # SAFETY CHECK
        # --------------------------------------------------
        if sign not in (+1, -1):
            raise RuntimeError(
                "AddNDigitEntity received invalid sign. "
                "SignRouterEntity must resolve sign first."
            )

        max_len = max(len(digits_a), len(digits_b))
        carry = 0
        result_digits = []

        for i in range(max_len):
            a = digits_a[i] if i < len(digits_a) else 0
            b = digits_b[i] if i < len(digits_b) else 0

            # -------------------------
            # Carry prediction
            # -------------------------
            state.arithmetic.digits_a = [a]
            state.arithmetic.digits_b = [b]
            state.arithmetic.carry = carry

            t0 = time.time()
            self.carry.forward(state)
            self.stats.log("CarryEntity.forward", time.time() - t0)

            carry_out = state.arithmetic.carry

            # -------------------------
            # Digit sum prediction
            # -------------------------
            t1 = time.time()
            self.add1.forward(state)
            self.stats.log("Add1DigitEntity.forward", time.time() - t1)

            raw_sum = state.arithmetic.sum
            sum_digit = (raw_sum + carry) % 10

            result_digits.append(sum_digit)

            state.trace.decisions.append(
                f"Digit {i}: a={a}, b={b}, carry_in={carry} "
                f"→ sum={sum_digit}, carry_out={carry_out}"
            )

            carry = carry_out

        # -------------------------
        # Final carry
        # -------------------------
        if carry == 1:
            result_digits.append(1)
            state.trace.decisions.append("Final carry appended")

        # -------------------------
        # Final magnitude (NO sign application)
        # -------------------------
        magnitude = int(
            "".join(map(str, reversed(result_digits)))
        )

        state.arithmetic.result_digits = result_digits
        state.arithmetic.result = magnitude
        state.arithmetic.sign = sign  # preserved

        state.control.done = True

        # -------------------------
        # Timing
        # -------------------------
        self.stats.log(
            "AddNDigitEntity.total",
            time.time() - start_total,
        )

        # --------------------------------------------------
        # 🔒 INVARIANT ENFORCEMENT (MANDATORY)
        # --------------------------------------------------
        ArithmeticInspector.check_invariants(state)

        if getattr(state.control, "final_report", False):
            self.stats.report()