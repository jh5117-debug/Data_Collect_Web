# Third-Party Versions

No third-party repository was cloned during this smoke implementation.

Current environment package inspection:

- Python: `3.13.9`
- torch: `2.9.1+cu128`
- transformers: installed
- huggingface_hub: installed
- modelscope: installed
- numpy: installed
- pandas: installed
- scipy: installed
- scikit-learn: installed
- matplotlib: installed
- pytest: installed
- openwakeword: not installed
- onnxruntime: not installed
- tensorboard: not installed

External model target:

- Qwen model: `Qwen/Qwen3-ASR-1.7B`
- Qwen weights were not downloaded in this task because CUDA was unavailable and network access is restricted in the current execution environment.

Official openWakeWord target:

- Upstream package/repository should be installed before treating Stage 1 metrics as scientific.
- The current smoke can run a clearly marked acoustic FFT fallback for pipeline wiring only.
