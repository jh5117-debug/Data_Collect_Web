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

    response = client.post("/api/admin/export")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "created"
    assert payload["file_name"].startswith("vigil_dataset_export_")
    download = client.get(payload["download_path"])
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/zip"
