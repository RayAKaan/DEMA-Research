from dataclasses import dataclass
from typing import Optional


@dataclass
class CalculusState:
    function: Optional[str] = None
    derivative: Optional[str] = None
    integral: Optional[str] = None
