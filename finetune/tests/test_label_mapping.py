from vigil_two_stage.export_parser import validate_and_map_clip


def test_p1_upload_creates_vigil_positive_labels():
    mapped, rejected = validate_and_map_clip(
        {
            "clip_id": "C1",
            "prompt_group": "P1_vigil_only",
            "transcript": "wrong ignored",
            "contains_vigil": True,
            "wake_intent": True,
            "is_negative": False,
        },
        ["visual"],
    )
    assert rejected is None
    assert mapped["transcript"] == "VIGIL"
    assert mapped["label"] == 1
    assert mapped["phrase_id"] == "vigil"


def test_p2_and_p3_require_exact_vigil():
    _, rejected2 = validate_and_map_clip({"clip_id": "C2", "prompt_group": "P2_phrase_plus_vigil", "transcript": "hello"}, ["visual"])
    _, rejected3 = validate_and_map_clip({"clip_id": "C3", "prompt_group": "P3_vigil_plus_phrase", "transcript": "hello"}, ["visual"])
    assert "positive_prompt_missing_exact_vigil" in rejected2["reasons"]
    assert "positive_prompt_missing_exact_vigil" in rejected3["reasons"]


def test_p4_rejects_exact_vigil_but_accepts_confusables():
    _, rejected = validate_and_map_clip({"clip_id": "C4", "prompt_group": "P4_negative", "transcript": "Vigil next"}, ["visual"])
    assert "negative_prompt_contains_exact_vigil" in rejected["reasons"]
    for word in ["visual", "digital", "individual", "visible", "vigilant"]:
        mapped, rejected = validate_and_map_clip(
            {
                "clip_id": word,
                "prompt_group": "P4_negative",
                "transcript": word,
                "contains_vigil": False,
                "wake_intent": False,
                "is_negative": True,
            },
            ["visual", "digital", "individual", "visible", "vigilant"],
        )
        assert rejected is None
        assert mapped["label"] == 0
