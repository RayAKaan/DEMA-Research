from dataclasses import dataclass, field, asdict

from state.control import ControlState
from state.arithmetic import ArithmeticState
from state.algebra import AlgebraState
from state.calculus import CalculusState
from state.logic import LogicState
from state.verification import VerificationState
from state.trace import TraceState


@dataclass
class GlobalState:
    control: ControlState = field(default_factory=ControlState)
    arithmetic: ArithmeticState = field(default_factory=ArithmeticState)
    algebra: AlgebraState = field(default_factory=AlgebraState)
    calculus: CalculusState = field(default_factory=CalculusState)
    logic: LogicState = field(default_factory=LogicState)
    verification: VerificationState = field(default_factory=VerificationState)
    trace: TraceState = field(default_factory=TraceState)

    def to_dict(self) -> dict:
        """Convert GlobalState into a plain dictionary (for logging / debugging)."""
        return asdict(self)

    def snapshot(self) -> dict:
        """Return a lightweight snapshot for trace logging."""
        return {
            "control": asdict(self.control),
            "arithmetic": asdict(self.arithmetic),
            "algebra": asdict(self.algebra),
            "calculus": asdict(self.calculus),
        }
