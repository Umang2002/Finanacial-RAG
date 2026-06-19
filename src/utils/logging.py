"""Shared rich-backed logger factory — every module logs through here instead of print().

WHAT: get_logger(name) returns a stdlib logging.Logger wired to a single
rich.Console via RichHandler.
WHY: CLAUDE.md mandates rich logging, no plain print(). One shared handler on
the root logger keeps output consistent (colors, timestamps, tracebacks)
across all 7 pipeline phases without each module reconfiguring logging.
"""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

# LEARN: one shared Console instance so every module's output goes to the same
# stream — avoids interleaved/duplicate formatting when multiple modules log.
_console = Console()


def _configure_root(level: int) -> None:
    """Attach a single RichHandler to the root logger, once.

    WHY: logging.basicConfig() is a no-op if the root logger already has
    handlers, but we check explicitly so re-importing this module in tests
    doesn't silently fail to configure logging.
    """
    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(level)
        root.addHandler(RichHandler(console=_console, rich_tracebacks=True, show_path=False))
        # LEARN: httpx/httpcore log every request at INFO — useful for
        # debugging but drowns out our own progress logs. Quiet them down;
        # set them to DEBUG explicitly if you need to see raw HTTP traffic.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module-scoped logger backed by rich.RichHandler.

    Args:
        name: usually `__name__` of the calling module.
        level: log level for this logger (default INFO).

    Returns:
        A configured logging.Logger — call .info()/.warning()/.debug() etc.
        instead of print().
    """
    _configure_root(level)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
