# Continuous ASR System Architecture

```text
Microphone
    |
    +--> continuously running Qwen ASR
    |       |
    |       +--> full doctor-patient transcript
    |             |
    |             +--> medical report
    |
    +--> parallel VIGIL wake-word branch
            |
            +--> Stage 1 openWakeWord
            |
            +--> Stage 2 verifier
                    |
                    +--> VIGIL detected
                          |
                          +--> enter assistant / VQA state
```

The wake-word detector does not start the medical transcript. The intended clinical ASR path is already listening continuously. The current prototype may execute operations sequentially offline, but the system design is continuous ASR plus a parallel wake-word branch. Current Stage 2 uses an additional Qwen audio-encoder forward for candidate windows; it is not zero additional Qwen compute.
