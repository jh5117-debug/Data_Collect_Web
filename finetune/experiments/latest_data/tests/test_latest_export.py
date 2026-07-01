from __future__ import annotations

from vigil_latest.export_download import api_url


def test_api_url_joins_backend_and_paths() -> None:
    assert api_url("https://example.test/", "/api/admin/export") == "https://example.test/api/admin/export"
    assert api_url("https://example.test", "api/admin/summary") == "https://example.test/api/admin/summary"
