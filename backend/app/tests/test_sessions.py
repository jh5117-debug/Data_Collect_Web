def test_create_and_submit_session(client):
    participant = client.post(
        "/api/participants",
        json={
            "english_native_speaker": "non_native_english_speaker",
            "recording_device_type": "headset_or_airpods",
        },
    ).json()

    response = client.post(
        "/api/sessions",
        json={"participant_id": participant["participant_id"], "batch_id": "vigil_batch_v0_1"},
    )

    assert response.status_code == 200
    assert response.json() == {"session_id": "S0001"}

    submit = client.post("/api/sessions/S0001/submit")
    assert submit.status_code == 200
    assert submit.json()["status"] == "submitted"
