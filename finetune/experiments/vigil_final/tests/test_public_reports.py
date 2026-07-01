from __future__ import annotations

import pytest

from vigil_final.privacy import assert_public_report_text


def test_no_generated_public_report_contains_names_or_emails_pattern():
    assert_public_report_text("Participant P001 aggregate only.")
    with pytest.raises(ValueError):
        assert_public_report_text("contact doctor@example.com")
