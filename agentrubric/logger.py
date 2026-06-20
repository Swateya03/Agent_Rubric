"""
logger.py — Centralised logging configuration for AgentRubric.

Every module that needs logging imports get_logger() from here.
Log level is controlled once at the application entry point (run_pipeline.py).

Usage in any module:
    from logger import get_logger
    logger = get_logger(__name__)
    logger.info("rubric_designer: retrieved %s", doc.name)
    logger.debug("Full state: %s", state)
    logger.warning("Hack detected: divergence=%.4f", divergence)
    logger.error("LLM call failed: %s", e)

Log levels:
    DEBUG   → verbose output, full state dumps, every node entry/exit
    INFO    → normal run output, one line per node completion
    WARNING → hack detections, quality flags, parse errors
    ERROR   → LLM failures, file not found, unrecoverable errors
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Get a named logger for a module.

    Uses the module __name__ as the logger name so log output shows
    which module produced each message.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured Logger instance. Adds a stdout StreamHandler if none exists.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(levelname)-8s %(name)s: %(message)s"
            )
        )
        logger.addHandler(handler)
        logger.propagate = False

    return logger


def configure_log_level(verbose: bool = False, quiet: bool = False) -> None:
    """Set the root log level for the entire application.

    Called once in run_pipeline.py main() based on CLI flags.

    Args:
        verbose: If True, set DEBUG level (all output including state dumps).
        quiet: If True, set WARNING level (only warnings and errors).
               If both are False, set INFO level (normal run output).
    """
    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.WARNING
    else:
        level = logging.INFO

    # Set level on the root agentrubric logger so all child loggers inherit it
    logging.getLogger("agentrubric").setLevel(level)

    # Also set on root for any loggers not under the agentrubric namespace
    logging.getLogger().setLevel(level)
