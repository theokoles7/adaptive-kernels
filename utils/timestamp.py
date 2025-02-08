"""Record timestamp for global time keeping."""

__all__ = ["TIMESTAMP"]

from datetime   import datetime

TIMESTAMP:  str =   datetime.now().strftime("%Y%m%d_%H%M%S")