import time


def _create_export_and_wait(client, attempts: int = 100):
    response = client.post("/api/admin/export")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"queued", "running", "completed", "completed_with_warnings"}
    assert payload["job_id"]

    for _ in range(attempts):
        status = client.get(f"/api/admin/export/jobs/{payload['job_id']}")
        assert status.status_code == 200
        job = status.json()
        if job["status"] in {"completed", "completed_with_warnings"}:
            return job
        if job["status"] == "failed":
            raise AssertionError(job["error_message"])
        time.sleep(0.05)
    raise AssertionError("export job did not finish")


def test_export_creation(client):
    client.post(
        "/api/participants",
        json={
            "english_native_speaker": "native_english_speaker",
            "recording_device_type": "smartphone",
        },
    )
    client.post(
        "/api/sessions",
        json={"participant_id": "P0001", "batch_id": "vigil_batch_v0_1"},
    )

    payload = _create_export_and_wait(client)

    assert payload["status"] == "completed"
    assert payload["file_name"].startswith("vigil_dataset_export_")
    assert payload["progress_percent"] == 100.0
    download = client.get(payload["download_path"])
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/zip"
