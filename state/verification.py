from dataclasses import dataclass, field
from typing import List


@dataclass
class VerificationState:
    verified: bool | None = None
    violations: List[str] = field(default_factory=list)
    checked_by: List[str] = field(default_factory=list)
