from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ArithmeticState:

    # Raw scalar inputs (optional, convenience only)
    a: Optional[int] = None
    b: Optional[int] = None

    # Magnitude representation (LSB-first)
    # Always NON-NEGATIVE digits
    digits_a: Optional[List[int]] = None
    digits_b: Optional[List[int]] = None

    # Sign of the arithmetic value
    # +1 for non-negative, -1 for negative
    sign: int = +1

    # Single-digit intermediates
    sum: Optional[int] = None
    carry: int = 0

    prod: Optional[int] = None
    diff: Optional[int] = None
    borrow: int = 0

    # Final outputs (magnitude only)
    result_digits: Optional[List[int]] = None
    result: Optional[int] = None