"""Unit tests for the output grounding guardrail."""

from src.guardrails.grounding import check_grounding


def test_grounded_answer_passes():
    context = "Duty exemption certificate applies to imported goods under 19 CFR 10.102"
    answer = "A duty exemption certificate applies to certain imported goods."
    grounded, overlap = check_grounding(answer, context)
    assert grounded is True
    assert overlap > 0.5


def test_ungrounded_answer_fails():
    context = "Duty exemption certificate applies to imported goods"
    answer = "Bananas and zebras enjoy spaceship holidays"
    grounded, overlap = check_grounding(answer, context)
    assert grounded is False


def test_empty_answer_is_treated_as_grounded():
    # Nothing substantive to verify -> don't falsely refuse.
    grounded, overlap = check_grounding("", "some context here")
    assert grounded is True
