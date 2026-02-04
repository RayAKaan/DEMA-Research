from dataclasses import dataclass
from typing import Optional


@dataclass
class LogicState:
    proposition: Optional[str] = None
    normalized: Optional[str] = None
    truth_value: Optional[bool] = None
