import time
import uuid
from contextlib import contextmanager

from src.observability.logging_config import get_logger
from src.observability.tracing import span

logger = get_logger("rag")


def new_request_id():
    """A short, unique id used to correlate every log line for one request."""
    return uuid.uuid4().hex[:12]


def log_event(event, request_id, **fields):
    """Emit a structured log line with a request_id and any extra fields."""
    logger.info(event, extra={"fields": {"request_id": request_id, **fields}})


@contextmanager
def timed_stage(stage, request_id):
    """Time a pipeline stage: emit a structured log AND an OpenTelemetry span.

    Usage:
        with timed_stage("retrieve", request_id):
            ...work...

    The log line gives you searchable per-stage timing; the span gives you a
    distributed trace (route/retrieve/generate) you can view in a tracing UI.
    If OpenTelemetry isn't installed, the span is a no-op and only the log fires.
    """
    start = time.perf_counter()
    with span(stage, {"request_id": request_id}) as current_span:
        try:
            yield
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            if current_span is not None:
                current_span.set_attribute("duration_ms", duration_ms)
            log_event("stage_complete", request_id, stage=stage, duration_ms=duration_ms)
