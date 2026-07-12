from contextlib import contextmanager

# OpenTelemetry is optional: if it isn't installed, tracing degrades to a no-op
# so the app and tests keep working. This is a good pattern in itself — the
# observability backend should never be a hard requirement for the code to run.
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on install
    _OTEL_AVAILABLE = False

_tracer = None


def get_tracer():
    """Return an initialized tracer, or None if OpenTelemetry isn't installed."""
    global _tracer
    if not _OTEL_AVAILABLE:
        return None
    if _tracer is None:
        provider = TracerProvider()
        # ConsoleSpanExporter prints spans to stdout for local dev. In production
        # you'd swap this for an OTLP exporter -> Azure Monitor / CloudWatch, with
        # no change to the calling code.
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("rag-logistics-assistant")
    return _tracer


@contextmanager
def span(name, attributes=None):
    """Context manager for an OpenTelemetry span.

    Always usable: yields a span when OTel is available (with the given
    attributes set), or None when it isn't — so callers work either way.
    """
    tracer = get_tracer()
    if tracer is None:
        yield None
        return

    with tracer.start_as_current_span(name) as current_span:
        if attributes:
            for key, value in attributes.items():
                current_span.set_attribute(key, value)
        yield current_span
