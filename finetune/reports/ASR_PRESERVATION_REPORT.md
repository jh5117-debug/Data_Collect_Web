# ASR Preservation Report

## Conclusion

Stage 2 uses frozen Qwen audio-encoder features. Qwen trainable parameters are zero and representative parameter checksums are unchanged. The two-stage trigger method does not update Qwen ASR weights. Therefore downstream general-ASR model behavior is represented by the unchanged Qwen baseline; wake-word detection metrics and LibriSpeech ASR metrics measure separate things.

## Frozen Qwen Evidence

Qwen model: `Qwen/Qwen3-ASR-1.7B`
Total Qwen parameters: `2038052480`
Feature extraction trainable Qwen parameters: `0`
Stage2 BCE trainable Qwen parameters: `0`
Stage2 BCE+SupCon trainable Qwen parameters: `0`
Corrected VIGIL baseline trainable Qwen parameters: `0`
Feature extraction checksums unchanged: `True`
Stage2 BCE checksums unchanged: `True`
Stage2 BCE+SupCon checksums unchanged: `True`

## Corrected LibriSpeech Qwen Baseline

Run: `/home/hj/Data_Collect_Web/finetune/benchmarks/asr/runs/20260624_185419_qwen3_asr_1_7b_fixed_text_extraction_baseline_full`
Successful predictions: `5559`
Failed predictions: `0`
Combined normalized WER: `0.02751646508258752`
test-clean normalized WER: `0.018411442483262326`
test-other normalized WER: `0.03666201784383776`
Raw WER: `0.9862560642019081`
Normalized CER: `0.009904616200632524`

## Corrected VIGIL Qwen Baseline

Run: `/home/hj/Data_Collect_Web/finetune/runs/20260624_075127_0fad4c7828149099_full/baseline_qwen_exact_clip_fixed_text_extraction`
n: `93`
Precision: `1.0`
Recall: `0.6862745098039216`
False-positive rate: `0.0`
F1: `0.813953488372093`
P1 recall: `0.6666666666666666`
P2 recall: `0.7272727272727273`
P3 recall: `0.6470588235294118`
P4 false-positive rate: `0.0`
Text extraction path: `['$[0].text']`
Qwen result type: `['qwen_asr.inference.qwen3_asr.ASRTranscription']`

## Invalid Historical Results

Old LibriSpeech run: `/home/hj/Data_Collect_Web/finetune/benchmarks/asr/runs/20260624_090118_qwen3_asr_1_7b_baseline_full`
Status: `invalid_for_scientific_reporting`
Reason: Old transcript extraction recursed into list[0] and then used str(result) for qwen_asr.inference.qwen3_asr.ASRTranscription, storing ASRTranscription(language=..., text=..., time_stamps=None) reprs instead of the .text transcript.
Correction commit: `538f646`
Old normalized WER `0.40069005613854497` is invalid and must not be cited as ASR performance.

## Metric Separation

- LibriSpeech WER measures Qwen general ASR.
- VIGIL recall/FPR measures wake-word detection.
- These metrics are separate; VIGIL trigger training does not fine-tune Qwen ASR weights.
