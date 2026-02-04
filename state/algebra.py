from dataclasses import dataclass
from typing import Optional


@dataclass
class AlgebraState:
    expr: Optional[str] = None
    simplified: Optional[str] = None
    expanded: Optional[str] = None
    factored: Optional[str] = None
    normalized: Optional[str] = None
    solution: Optional[str] = None
