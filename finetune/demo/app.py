from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gradio as gr

from inference import VigilInference
from model_loader import load_runtime
from plotting import stage1_score_plot


INFERENCE: VigilInference | None = None


def _format_seconds(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.3f}s"


def analyze_audio(audio_path: str | None, variant: str, transcript_after_trigger: bool):
    if INFERENCE is None:
        raise gr.Error("model is not loaded")
    if not audio_path:
        raise gr.Error("audio input is required")
    try:
        result = INFERENCE.analyze(
            Path(audio_path),
            variant_label=variant,
            run_transcript_after_trigger=transcript_after_trigger,
        )
    except Exception as exc:
        raise gr.Error(f"{type(exc).__name__}: {exc}") from exc
    winning = result["winning_window"] or {}
    timings = result["timings"]
    plot = stage1_score_plot(result["stage1_scores"], result["theta_1"])
    return (
        result["result_text"],
        result["variant"],
        f"{result['stage1_score']:.6f}",
        f"{result['theta_1']:.6f}",
        "" if result["stage2_score"] is None else f"{result['stage2_score']:.6f}",
        f"{result['theta_2']:.6f}",
        "" if not winning else f"#{winning['index']} {winning['start_sec']:.2f}-{winning['end_sec']:.2f}s",
        _format_seconds(timings["stage1_latency_sec"]),
        _format_seconds(timings["qwen_encoder_latency_sec"]),
        _format_seconds(timings["stage2_head_latency_sec"]),
        _format_seconds(timings["asr_transcript_latency_sec"]),
        _format_seconds(timings["total_latency_sec"]),
        result["transcript"],
        result["window_table"],
        plot,
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="VIGIL Recorder Demo") as app:
        gr.Markdown("# VIGIL Recorder")
        with gr.Row():
            audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio")
            with gr.Column():
                variant = gr.Dropdown(
                    choices=["Validation-selected", "BCE", "BCE + SupCon"],
                    value="Validation-selected",
                    label="Model variant",
                )
                transcript = gr.Checkbox(value=True, label="Run Qwen transcript after accepted trigger")
                analyze = gr.Button("Analyze", variant="primary")
        with gr.Row():
            result_text = gr.Textbox(label="Result", interactive=False)
            selected_variant = gr.Textbox(label="Variant", interactive=False)
            winning_window = gr.Textbox(label="Window", interactive=False)
        with gr.Row():
            stage1_score = gr.Textbox(label="Stage 1 score", interactive=False)
            theta1 = gr.Textbox(label="Theta 1", interactive=False)
            stage2_score = gr.Textbox(label="Stage 2 score", interactive=False)
            theta2 = gr.Textbox(label="Theta 2", interactive=False)
        with gr.Row():
            stage1_latency = gr.Textbox(label="Stage 1 latency", interactive=False)
            qwen_latency = gr.Textbox(label="Qwen encoder latency", interactive=False)
            stage2_latency = gr.Textbox(label="Stage 2 latency", interactive=False)
            asr_latency = gr.Textbox(label="ASR latency", interactive=False)
            total_latency = gr.Textbox(label="Total latency", interactive=False)
        transcript_out = gr.Textbox(label="Transcript", lines=4, interactive=False)
        window_table = gr.Dataframe(label="Windows", interactive=False)
        plot = gr.Plot(label="Stage 1")
        analyze.click(
            analyze_audio,
            inputs=[audio, variant, transcript],
            outputs=[
                result_text,
                selected_variant,
                stage1_score,
                theta1,
                stage2_score,
                theta2,
                winning_window,
                stage1_latency,
                qwen_latency,
                stage2_latency,
                asr_latency,
                total_latency,
                transcript_out,
                window_table,
                plot,
            ],
        )
    return app


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    global INFERENCE
    runtime = load_runtime(args.run_dir)
    INFERENCE = VigilInference(runtime)
    app = build_app()
    app.queue()
    app.launch(server_name=args.host, server_port=args.port, share=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

