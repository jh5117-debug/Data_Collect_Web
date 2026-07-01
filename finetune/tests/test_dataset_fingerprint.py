from finetune.scripts.prepare_dataset import dataset_fingerprint


def test_repeated_dataset_preparation_produces_identical_fingerprint():
    cfg = {"seed": 20260620, "audio": {"sample_rate": 16000}, "data": {"speaker_split": True}}
    assert dataset_fingerprint("abc", cfg) == dataset_fingerprint("abc", cfg)
