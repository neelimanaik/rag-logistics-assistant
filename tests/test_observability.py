"""Unit tests for the observability helpers."""

import logging

from src.observability.instrument import log_event, new_request_id, timed_stage
from src.observability.tracing import span


def test_new_request_id_is_hex12():
    rid = new_request_id()
    assert len(rid) == 12
    int(rid, 16)  # parses as hex


def test_log_event_attaches_request_id_and_fields(caplog):
    with caplog.at_level(logging.INFO):
        log_event("request_start", "rid123", question_chars=10)
    records = [r for r in caplog.records if r.getMessage() == "request_start"]
    assert records
    assert records[0].fields["request_id"] == "rid123"
    assert records[0].fields["question_chars"] == 10


def test_timed_stage_logs_stage_and_duration(caplog):
    with caplog.at_level(logging.INFO):
        with timed_stage("retrieve", "abc123"):
            pass
    records = [
        r for r in caplog.records if getattr(r, "fields", {}).get("stage") == "retrieve"
    ]
    assert records
    assert records[0].fields["request_id"] == "abc123"
    assert "duration_ms" in records[0].fields


def test_span_is_a_usable_context_manager():
    # Works whether or not OpenTelemetry is installed (no-op yields None).
    with span("unit_test_span", {"request_id": "r"}) as s:
        assert s is None or s is not None  # just must not raise
