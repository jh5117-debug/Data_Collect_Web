#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-vigil-two-stage}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

if command -v conda >/dev/null 2>&1; then
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
  conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
  conda run -n "${ENV_NAME}" python -m pip install numpy scipy pandas scikit-learn pyyaml tqdm matplotlib pytest torch tensorboard huggingface_hub transformers
else
  python3 -m venv "finetune/.venv"
  finetune/.venv/bin/python -m pip install --upgrade pip
  finetune/.venv/bin/python -m pip install numpy scipy pandas scikit-learn pyyaml tqdm matplotlib pytest torch tensorboard huggingface_hub transformers
fi

echo "Bootstrap complete. Install openWakeWord and Qwen dependencies in this environment when network/GPU access is available."
