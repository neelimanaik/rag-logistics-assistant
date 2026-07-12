import json
import logging
import sys

_configured = False


class JsonFormatter(logging.Formatter):
    """Render each log record as a single JSON line.

    Structured (machine-readable) logs are the foundation of observability: you
    can search, filter, and aggregate them. Anything passed via
    `extra={"fields": {...}}` is merged into the JSON so callers can attach a
    request_id, stage name, duration, etc.
    """

    def format(self, record):
        payload = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        return json.dumps(payload)


def configure_logging(level=logging.INFO):
    """Attach a JSON handler to the root logger, once.

    We intentionally do NOT clear existing handlers, so test tooling (pytest's
    caplog) keeps working alongside our handler.
    """
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.addHandler(handler)
    if root.level == logging.NOTSET or root.level > level:
        root.setLevel(level)
    _configured = True


def get_logger(name):
    configure_logging()
    return logging.getLogger(name)
