from fractions import Fraction
from typing import Annotated, Any, Optional

from pydantic import BeforeValidator


def parse_fraction_string(v: Any) -> Any:
    """Logic to convert string fractions to floats before Pydantic validates."""
    if isinstance(v, str):
        try:
            return float(Fraction(v))
        except (ValueError, ZeroDivisionError):
            raise ValueError(f"Invalid fraction string: {v}")
    return v


# Create a custom type:
# It says: "The final type is float | None, but run this parser first"
FlexFloat = Annotated[Optional[float], BeforeValidator(parse_fraction_string)]
