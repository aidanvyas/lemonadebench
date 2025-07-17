from __future__ import annotations

from decimal import Decimal, getcontext

# Set global precision for all Decimal operations
getcontext().prec = 28


def to_decimal(value: int | float | str | Decimal) -> Decimal:
    """Convert various numeric types to ``Decimal`` safely."""
    return Decimal(str(value))
