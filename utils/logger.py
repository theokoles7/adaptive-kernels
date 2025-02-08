"""Logging utilities."""

__all__ = ["LOGGER"]

from logging            import getLogger, Formatter, Logger, StreamHandler
from logging.handlers   import RotatingFileHandler
from os                 import makedirs
from os.path            import dirname
from sys                import stdout

from utils.arguments    import ARGS
from utils.timestamp    import TIMESTAMP

# Match command being executed
match ARGS.cmd:
    
    # For jobs...
    case "run-job":
        
        _log_path_: str =   f"{ARGS.logging_path}/jobs/{ARGS.dataset}/{ARGS.model}/{f"{ARGS.kernel}-{ARGS.kernel_group}" if ARGS.kernel is not None else "control"}/{TIMESTAMP}.log"
        
    # For experiments...
    case "run-experiment":
        
        _log_path_: str =   f"{ARGS.logging_path}/experiments/{TIMESTAMP}.log"

# Ensure that logging path exists
makedirs(name = dirname(p = _log_path_), exist_ok = True)

# Initialize logger
LOGGER:         Logger =                getLogger("adaptive-kernel")

# Set logging level
LOGGER.setLevel(ARGS.logging_level)

# Define console handler
stdout_handler: StreamHandler =         StreamHandler(stdout)
stdout_handler.setFormatter(Formatter("%(levelname)s | %(name)s | %(message)s"))
LOGGER.addHandler(stdout_handler)

# Define file handler
file_handler:   RotatingFileHandler =   RotatingFileHandler(_log_path_, maxBytes = 1048576)
file_handler.setFormatter(Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
LOGGER.addHandler(file_handler)