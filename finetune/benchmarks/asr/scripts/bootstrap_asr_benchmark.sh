#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/hj/Data_Collect_Web"
ENV_PY="/home/hj/miniconda/envs/vigil-two-stage/bin/python"

if [[ ! -x "$ENV_PY" ]]; then
  echo "Missing Python environment: $ENV_PY" >&2
  exit 1
fi

cd "$PROJECT_ROOT"

set +e
check_output="$("$ENV_PY" - <<'PY'
import importlib.util
import sys

required = {
    "torch": "torch",
    "torchaudio": "torchaudio",
    "qwen_asr": "qwen_asr",
    "transformers": "transformers",
    "huggingface_hub": "huggingface_hub",
    "soundfile": "soundfile",
    "numpy": "numpy",
    "tqdm": "tqdm",
    "yaml": "pyyaml",
    "pytest": "pytest",
}

missing = []
for module, package in required.items():
    if importlib.util.find_spec(module) is None:
        missing.append(package)
if missing:
    print(" ".join(sorted(set(missing))))
    sys.exit(2)
print("all core packages present")
PY
)"
status=$?
set -e

if [[ "$status" -eq 2 ]]; then
  missing="$check_output"
  echo "Installing missing lightweight benchmark dependencies: $missing"
  "$ENV_PY" -m pip install $missing
elif [[ "$status" -ne 0 ]]; then
  echo "$check_output" >&2
  exit "$status"
else
  echo "$check_output"
fi

"$ENV_PY" - <<'PY'
import importlib.util
import torch

print("python ok")
print("torch", torch.__version__)
print("cuda build", torch.version.cuda)
print("cuda available in this shell", torch.cuda.is_available())
for module in ("qwen_asr", "transformers", "huggingface_hub", "soundfile", "yaml", "pytest"):
    if importlib.util.find_spec(module) is None:
        raise SystemExit(f"missing required module after bootstrap: {module}")
print("ASR benchmark environment is ready")
PY
