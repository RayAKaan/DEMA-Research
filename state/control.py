from dataclasses import dataclass


@dataclass
class ControlState:
    goal: str | None = None
    done: bool = False
    step: int = 0
    confidence: float | None = None
