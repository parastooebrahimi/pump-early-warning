"""Early-warning analytics for industrial pump sensor data."""

from .config import Config
from .pipeline import run

__all__ = ["Config", "run"]
