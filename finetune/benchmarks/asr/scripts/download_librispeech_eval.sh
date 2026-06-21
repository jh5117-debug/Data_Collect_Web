#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/hj/Data_Collect_Web"
BENCH_ROOT="$PROJECT_ROOT/finetune/benchmarks/asr"
ENV_PY="/home/hj/miniconda/envs/vigil-two-stage/bin/python"
DOWNLOAD_DIR="$BENCH_ROOT/downloads"
DATA_PARENT="$BENCH_ROOT/data"
LIBRISPEECH_ROOT="$DATA_PARENT/LibriSpeech"

mkdir -p "$DOWNLOAD_DIR" "$DATA_PARENT" "$LIBRISPEECH_ROOT" "$BENCH_ROOT/reports" "$BENCH_ROOT/manifests"
cd "$PROJECT_ROOT"

download_one() {
  local url="$1"
  local name="$2"
  local archive="$DOWNLOAD_DIR/$name"
  if [[ -s "$archive" ]]; then
    echo "Archive already exists: $archive"
    return
  fi
  local partial="$archive.partial"
  echo "Downloading $url"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 5 --continue-at - -o "$partial" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget --continue -O "$partial" "$url"
  else
    echo "Neither curl nor wget is available" >&2
    exit 1
  fi
  mv "$partial" "$archive"
}

extract_one() {
  local archive="$1"
  local split="$2"
  if [[ -d "$LIBRISPEECH_ROOT/$split" ]]; then
    echo "Extracted split already exists: $LIBRISPEECH_ROOT/$split"
    return
  fi
  echo "Extracting $archive"
  tar -xzf "$archive" -C "$DATA_PARENT"
}

download_one "https://www.openslr.org/resources/12/test-clean.tar.gz" "test-clean.tar.gz"
download_one "https://www.openslr.org/resources/12/test-other.tar.gz" "test-other.tar.gz"

extract_one "$DOWNLOAD_DIR/test-clean.tar.gz" "test-clean"
extract_one "$DOWNLOAD_DIR/test-other.tar.gz" "test-other"

"$ENV_PY" "$BENCH_ROOT/scripts/prepare_librispeech_manifest.py" \
  --root "$LIBRISPEECH_ROOT" \
  --manifest-dir "$BENCH_ROOT/manifests" \
  --reports-dir "$BENCH_ROOT/reports" \
  --expected-counts \
  --validate-audio
