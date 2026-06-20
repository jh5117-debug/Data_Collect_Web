DATA_ZIP ?= /home/hj/Data_Collect_Web/finetune/data/vigil_dataset_export_20260620_020617.zip

.PHONY: bootstrap prepare smoke full test

bootstrap:
	bash finetune/scripts/bootstrap_env.sh

prepare:
	PYTHONPATH=finetune/src python finetune/scripts/prepare_dataset.py "$(DATA_ZIP)" --config finetune/configs/smoke.yaml --output-root finetune/data/processed

smoke:
	bash finetune/scripts/run_smoke.sh "$(DATA_ZIP)"

full:
	bash finetune/scripts/run_full.sh "$(DATA_ZIP)"

test:
	PYTHONPATH=finetune/src:. pytest -q finetune/tests
