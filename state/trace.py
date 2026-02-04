from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class TraceState:
    history: List[Dict[str, Any]] = field(default_factory=list)
    entities_used: List[str] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
