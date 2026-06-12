def test_prompts_load_without_calibration(client):
    response = client.get("/api/prompts")

    assert response.status_code == 200
    prompts = response.json()
    assert len(prompts) == 5
    assert all(prompt["prompt_id"] != "CALIBRATION" for prompt in prompts)
    assert [prompt["prompt_id"] for prompt in prompts] == [
        "POS_SINGLE_001",
        "POS_SINGLE_002",
        "POS_SINGLE_003",
        "POS_SINGLE_004",
        "POS_SINGLE_005",
    ]
