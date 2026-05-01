from __future__ import annotations

from decimal import Decimal


def to_decimal(value: int | float | str | Decimal) -> Decimal:
    """Convert various numeric types to Decimal without float encoding errors.

    By converting to string first, we avoid float precision issues:
    - Decimal(0.1) → Decimal('0.1000000000000000055...')  [wrong]
    - Decimal(str(0.1)) → Decimal('0.1')  [correct]
    """
    return Decimal(str(value))
