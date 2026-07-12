"""Custom logger for project."""

import argparse
import copy
import datetime as dt
import io
import json
import logging
import logging.config
import sys
from pathlib import Path
from types import TracebackType
from typing import Any, ClassVar

# for colored logs
import colorama
import yaml
from typing_extensions import override  # 2026-04-22: typing.override is 3.12+; pinned to 3.11

my_logger = logging.getLogger(name=__name__)

_CONFIG_PATH = Path(__file__).parent / "logger_config.yaml"

LOG_RECORD_BUILTIN_ATTRS: frozenset[str] = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)


class ColorFormatter(logging.Formatter):
    """Color formatter for logs."""

    LOG_COLORS: ClassVar[dict[int, str]] = {
        logging.DEBUG: colorama.Fore.GREEN,
        logging.INFO: colorama.Fore.BLUE,
        logging.WARNING: colorama.Fore.YELLOW,
        logging.ERROR: colorama.Fore.RED,
        logging.CRITICAL: colorama.Back.RED,
    }

    DEFAULT_FMT = "%(levelname)-8s %(name)s: %(message)s"

    def __init__(
        self,
        *,
        fmt: str | None = None,
        datefmt: str | None = None,
    ) -> None:
        """Initialize ColorFormatter object."""
        super().__init__(fmt=fmt or self.DEFAULT_FMT, datefmt=datefmt)

    @override
    def format(self, record: logging.LogRecord) -> str:
        """Format log record."""
        # if the corresponding logger has children, they may receive modified
        # record, so we want to keep it intact
        new_record = copy.copy(record)
        if new_record.levelno in self.LOG_COLORS:
            new_record.levelname = (
                f"{self.LOG_COLORS[new_record.levelno]}"
                f"{new_record.levelname}"
                f"{colorama.Style.RESET_ALL}"
            )
        return super().format(new_record)


class MyJSONFormatter(logging.Formatter):
    """Formatter for json logs."""

    def __init__(
        self,
        *,
        fmt_keys: dict[str, str] | None = None,
    ) -> None:
        """Initialize json log formatter."""
        super().__init__()
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}

    @override
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        message = self._prepare_log_dict(record)
        return json.dumps(message, default=str)

    def _prepare_log_dict(self, record: logging.LogRecord) -> dict[str, str | Any]:
        always_fields = {
            "message": record.getMessage(),
            "timestamp": dt.datetime.fromtimestamp(record.created, tz=dt.UTC).isoformat(),
        }
        if record.exc_info is not None:
            always_fields["exc_info"] = self.formatException(record.exc_info)

        if record.stack_info is not None:
            always_fields["stack_info"] = self.formatStack(record.stack_info)

        message = {
            key: msg_val
            if (msg_val := always_fields.pop(val, None)) is not None
            else getattr(record, val)
            for key, val in self.fmt_keys.items()
        }
        message.update(always_fields)
        message.update(
            {
                key: val
                for key, val in record.__dict__.items()
                if key not in LOG_RECORD_BUILTIN_ATTRS
            }
        )

        return message


class NonErrorFilter(logging.Filter):
    """Filter to exclude error logs (only INFO and DEBUG will pass through)."""

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        """Pass DEBUG and INFO records only."""
        # 2026-04-22: stdlib widened this to `bool | LogRecord` in 3.12; we're pinned to 3.11.
        return record.levelno <= logging.INFO


def _force_utf8_console() -> None:
    """Make console logging UTF-8 safe.

    Windows consoles default to cp1252, which can't encode the ✓/✗ marks and
    em-dash used in log messages (raising UnicodeEncodeError inside
    ``logging.Handler.emit`` — the batch summary then never prints). Switch
    stdout/stderr to UTF-8 with ``backslashreplace`` so non-cp1252 glyphs never
    crash a handler; a no-op on already-UTF-8 platforms.
    """
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")


def setup_logger() -> None:
    """Configure the logger from YAML config file."""
    _force_utf8_console()
    Path("logs").mkdir(exist_ok=True)
    with _CONFIG_PATH.open() as f_in:
        config: Any = yaml.safe_load(f_in)
    logging.config.dictConfig(config)


def _excepthook(
    exc_type: type[BaseException],
    exc: BaseException,
    tb: TracebackType | None,
) -> None:
    """Route uncaught exceptions through my_logger; preserve KeyboardInterrupt."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc, tb)
        return
    my_logger.critical("Unhandled exception", exc_info=(exc_type, exc, tb))


def install_excepthook() -> None:
    """Install the uncaught-exception handler on sys.excepthook."""
    sys.excepthook = _excepthook


def initialize_logger(args: argparse.Namespace) -> None:
    """Set up logger, install excepthook, apply runtime overrides from parsed CLI args.

    Args:
        args: argparse.Namespace with optional boolean attribute ``debug``.

    """
    setup_logger()
    install_excepthook()
    if getattr(args, "debug", False):
        logging.getLogger().setLevel(logging.DEBUG)
