# Environment Audit

Collected before implementing the two-stage smoke package.

## Repository

- Project root: `/home/hj/Data_Collect_Web`
- Initial branch: `main`
- Initial commit: `227577d6451db314f8959835852f52c3801579a8`
- Remote: `origin git@github.com:jh5117-debug/Data_Collect_Web.git`
- Initial uncommitted change: `docs/VIGIL_Recorder_Participant_Guide.docx` was untracked before this task and was not touched.

## Host

- Working user: `hj`
- Hostname: `hal-9000`
- Audit time: `2026-06-20T04:30:46+02:00`
- Python: `Python 3.13.9`
- Conda: present at `/home/hj/miniconda/bin/conda`, with a sqlite plugin warning during inspection
- Mamba: present at `/home/hj/miniconda/bin/mamba`
- ffmpeg: present, version `4.4.2-0ubuntu0.22.04.1`
- gcc: present, version `11.4.0`
- PyTorch in current environment: `2.9.1+cu128`
- `torch.cuda.is_available()`: `False`
- `nvidia-smi`: failed because the NVIDIA driver was not visible
- Slurm: `sbatch` not found, `sinfo` not found
- `SLURM_JOB_ID`: empty

## Disk

- `/home/hj`: about `1.0T` total, `549G` free during audit
- `/tmp`: about `1.8T` total, `262G` free during audit

## Source ZIP

- Path: `/home/hj/Data_Collect_Web/finetune/data/vigil_dataset_export_20260620_020617.zip`
- Size: about `61M`
- SHA-256: `f0808053b27d2c5d0af834df5ecc3662af7dd47fbf3984691da27bfbd8b6628d`
- `unzip -t`: passed
- Metadata clip rows: `403`
- Sessions: `13`
- Accounts: `10`
- Canonical `audio_raw/` files: `403`
- Duplicate `raw_audio/` view files: `806`
- Duplicate `by_prompt_group/` audio files: `403`

## Prompt Group Counts

- `P1_vigil_only`: `31`
- `P2_phrase_plus_vigil`: `111`
- `P3_vigil_plus_phrase`: `76`
- `P4_negative`: `185`

## Dependency Availability In Current Environment

- numpy: available
- pandas: available
- PyYAML: available
- scipy: available
- scikit-learn: available
- matplotlib: available
- pytest: available
- torch: available
- transformers: available
- huggingface_hub: available
- modelscope: available
- soundfile: not installed
- torchaudio: not installed
- librosa: not installed
- openwakeword: not installed
- onnxruntime: not installed
- tensorboard: not installed

## Blockers For Full Scientific Run

- No visible CUDA GPU in the current shell.
- No Slurm command in PATH.
- Official openWakeWord is not installed.
- Qwen3-ASR-1.7B weights were not found in the inspected user caches.
