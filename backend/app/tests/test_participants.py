def test_create_participant(client):
    response = client.post(
        "/api/participants",
        json={
            "english_native_speaker": "native_english_speaker",
            "recording_device_type": "laptop_builtin_microphone",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"participant_id": "P0001"}
