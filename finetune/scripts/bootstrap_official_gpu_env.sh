#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/home/hj/Data_Collect_Web"
ENV_NAME="${ENV_NAME:-vigil-two-stage}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
GPU_INDEX="${GPU_INDEX:-6}"

cd "${ROOT}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is required for the official GPU smoke environment on HAL." >&2
  echo "Please run this from a normal HAL SSH shell where conda is available." >&2
  exit 2
fi

print_conda_tos_help() {
  cat >&2 <<'EOF'
ERROR: Conda requires you to accept the Anaconda channel Terms of Service first.

Run these two commands from the normal HAL SSH shell, then rerun this bootstrap:

  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

EOF
}

CONDA_ERR="$(mktemp)"
if ! conda env list > /tmp/vigil_conda_envs.txt 2>"${CONDA_ERR}"; then
  if grep -q "CondaToSNonInteractiveError" "${CONDA_ERR}"; then
    print_conda_tos_help
    exit 2
  fi
  cat "${CONDA_ERR}" >&2
  exit 2
fi

if ! awk '{print $1}' /tmp/vigil_conda_envs.txt | grep -qx "${ENV_NAME}"; then
  if ! conda create -y --solver=classic -n "${ENV_NAME}" "python=${PYTHON_VERSION}" 2>"${CONDA_ERR}"; then
    if grep -q "CondaToSNonInteractiveError" "${CONDA_ERR}"; then
      print_conda_tos_help
      exit 2
    fi
    cat "${CONDA_ERR}" >&2
    exit 2
  fi
fi
rm -f "${CONDA_ERR}" /tmp/vigil_conda_envs.txt

conda run -n "${ENV_NAME}" python -m pip install --upgrade pip setuptools wheel

# CUDA 13.2 drivers can run CUDA 12.8 PyTorch wheels. Keep this explicit so the
# env does not accidentally receive a CPU-only torch build.
conda run -n "${ENV_NAME}" python -m pip install --upgrade \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch torchaudio

conda run -n "${ENV_NAME}" python -m pip install --upgrade \
  numpy scipy pandas scikit-learn pyyaml tqdm matplotlib pytest tensorboard \
  soundfile librosa safetensors sentencepiece accelerate huggingface_hub \
  transformers openwakeword qwen-asr

conda run -n "${ENV_NAME}" python - <<'PY'
import importlib.util
import torch

missing = [name for name in ("openwakeword", "qwen_asr") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"missing required packages after install: {missing}")

print("python packages ok")
print("torch", torch.__version__, "cuda", torch.version.cuda)
PY

cat <<EOF
Official GPU smoke environment is ready.

Run the strict one-GPU smoke from normal HAL SSH with:

  cd ${ROOT}
  tmux new -d -s vigil_smoke_3090 "conda run -n ${ENV_NAME} bash -lc 'cd ${ROOT} && bash finetune/scripts/run_official_smoke_local_3090.sh ${GPU_INDEX}'"

Monitor it with:

  tmux attach -t vigil_smoke_3090

or:

  tail -f ${ROOT}/finetune/logs/official_smoke_3090_*_gpu${GPU_INDEX}.log
EOF
