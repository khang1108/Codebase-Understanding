"""Application-wide logging configuration for codebase-rag.

Provides a single ``setup_logging()`` function that must be called once at
application startup (``main.py``, CLI entry point, or Streamlit app entry).
Never call this inside library modules; use ``logging.getLogger(__name__)``
there instead.

Two output modes are supported:

- **Console** (default / development): human-readable lines with timestamp,
    log level, logger name, and line number.
- **JSON** (production): one JSON object per line, suitable for log
    aggregators such as Datadog, CloudWatch, or Loki.

Example::

    # main.py or scripts/index_repo.py
    from codebase_rag.core.logging import setup_logging
    from codebase_rag.core.config import Settings

    settings = Settings()
    setup_logging(level=settings.log_level, json_logs=settings.log_json)

    How every other file in the project uses it

    # any module — just this, nothing else
    import logging

    logger = logging.getLogger(__name__)

    # then inside methods:
    logger.info("Upserting %d chunks", len(chunks))
    logger.debug("Query vector dim=%d", len(vector))
    logger.error("Collection not found: %s", self.collection_name)
"""

from __future__ import annotations

import json
import logging
import logging.config
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Third-party libraries that produce excessive DEBUG / INFO noise.
# Raised to WARNING so our own application logs stay visible.
_NOISY_LOGGERS: tuple[str, ...] = (
    "sentence_transformers",
    "qdrant_client",
    "httpx",
    "httpcore",
    "urllib3",
    "transformers",
    "torch",
    "filelock",
    "huggingface_hub",
)

_VALID_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


class _JsonFormatter(logging.Formatter):
    """Formats each log record as a single-line JSON object.

    Intended for production environments where logs are shipped to an
    aggregator (Datadog, CloudWatch, Loki, etc.). Exception tracebacks
    are included as a string under the ``"exception"`` key.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_dict_config(level: str, json_logs: bool) -> dict[str, Any]:
    """Construct a ``logging.config.dictConfig``-compatible configuration.

    Args:
        level:     Root log level string (e.g. ``"INFO"``). Must be
            upper-cased before passing.
        json_logs: Select JSON formatter when ``True``, console formatter
            when ``False``.

    Returns:
        A configuration dict accepted by ``logging.config.dictConfig``.
    """
    active_formatter = "json" if json_logs else "console"

    return {
        "version": 1,
        # False = preserve loggers created before setup_logging() was called.
        # True would silently break third-party library loggers.
        "disable_existing_loggers": False,
        "formatters": {
            "console": {
                "format": _CONSOLE_FORMAT,
                "datefmt": _DATE_FORMAT,
            },
            "json": {
                # "()" is dictConfig syntax for instantiating a custom class.
                "()": _JsonFormatter,
        },},
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": active_formatter,
        },},
        "root": {
            "level": level,
            "handlers": ["stdout"],
    },}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def setup_logging(
    level: str = "INFO",
    json_logs: bool = False,
    noisy_log_level: str = "WARNING",
) -> None:
    """Configure application-wide logging. Call once at startup.

    Applies a ``dictConfig``-based configuration to the root logger, then
    raises the log level of known noisy third-party libraries so they do
    not pollute the output.

    Args:
        level:           Root log level. Accepted values (case-insensitive):
            ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``,
            ``CRITICAL``. Defaults to ``"INFO"``.
        json_logs:       Emit JSON-structured logs when ``True`` (production).
            Emit human-readable console logs when ``False``
            (development). Defaults to ``False``.
        noisy_log_level: Log level applied to verbose third-party libraries.
            Defaults to ``"WARNING"``.

    Raises:
        ValueError: If ``level`` or ``noisy_log_level`` is not a recognised
            log-level string.

    Example::

        setup_logging(level="DEBUG", json_logs=False)  # local dev
        setup_logging(level="INFO",  json_logs=True)   # production
    """
    normalised_level = level.upper()
    normalised_noisy = noisy_log_level.upper()

    if normalised_level not in _VALID_LEVELS:
        raise ValueError(
            f"Unrecognised log level {level!r}. Valid options: {sorted(_VALID_LEVELS)}"
        )
    if normalised_noisy not in _VALID_LEVELS:
        raise ValueError(
            f"Unrecognised noisy_log_level {noisy_log_level!r}. Valid options: {sorted(_VALID_LEVELS)}"
        )

    logging.config.dictConfig(_build_dict_config(normalised_level, json_logs))

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(normalised_noisy)

    logging.getLogger(__name__).debug(
        "Logging initialised (level=%s, json=%s)", normalised_level, json_logs
    )