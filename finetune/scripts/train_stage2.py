#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from vigil_two_stage.losses import bce_with_logits_loss, supervised_contrastive_loss
from vigil_two_stage.metrics import binary_metrics
from vigil_two_stage.qwen_audio_adapter import checksum_representative_parameters
from vigil_two_stage.stage2_model import QwenVerifierHead
from vigil_two_stage.thresholds import select_recall_first_threshold
from vigil_two_stage.utils import ensure_dir, read_json, read_jsonl, seed_everything, write_json, write_jsonl


class QwenFeatureDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        data = np.load(row["feature_path"])
        if "features" in data:
            arr = data["features"]
        elif "hidden_states" in data:
            arr = data["hidden_states"]
        else:
            arr = data[data.files[0]]
        return arr.astype(np.float32), int(row["label"]), row["phrase_id"], row


def collate(batch):
    arrays, labels, phrase_ids, rows = zip(*batch)
    lengths = torch.tensor([arr.shape[0] for arr in arrays], dtype=torch.long)
    max_len = int(lengths.max().item())
    dim = arrays[0].shape[1]
    hidden = torch.zeros(len(arrays), max_len, dim, dtype=torch.float32)
    mask = torch.zeros(len(arrays), max_len, dtype=torch.bool)
    for i, arr in enumerate(arrays):
        hidden[i, : arr.shape[0]] = torch.from_numpy(arr)
        mask[i, : arr.shape[0]] = True
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    return hidden, mask, labels_tensor, list(phrase_ids), rows


def predict(model, rows, device):
    if not rows:
        return []
    loader = DataLoader(QwenFeatureDataset(rows), batch_size=16, shuffle=False, collate_fn=collate)
    out = []
    model.eval()
    with torch.no_grad():
        for hidden, mask, labels, phrase_ids, batch_rows in loader:
            result = model(hidden.to(device), mask.to(device))
            scores = torch.sigmoid(result["logit"]).detach().cpu().numpy().tolist()
            for row, score in zip(batch_rows, scores):
                pred = {
                    k: row[k]
                    for k in ("clip_id", "speaker_id", "session_id", "prompt_group", "transcript", "label", "phrase_id", "split")
                }
                if "window_index" in row:
                    pred["window_index"] = row["window_index"]
                pred["stage2_score"] = float(score)
                out.append(pred)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--variant", choices=["bce", "bce_supcon"], default="bce")
    parser.add_argument("--allow-skip", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed_everything(int(config["seed"]))
    out_dir = ensure_dir(Path(args.run_dir) / ("stage2_" + args.variant))
    feature_manifest = Path(args.run_dir) / "stage2_qwen_features" / "qwen_features_manifest.jsonl"
    if not feature_manifest.exists():
        status = {
            "status": "skipped",
            "variant": args.variant,
            "reason": "Qwen encoder feature manifest is unavailable; verifier training was not run.",
            "model_name": config["stage2"]["model_name"],
            "qwen_parameters_modified": False,
        }
        write_json(out_dir / "metrics.json", status)
        write_json(out_dir / "frozen_qwen_integrity.json", {"status": "not_run", "qwen_parameters_modified": False, "reason": status["reason"]})
        (out_dir / "report.md").write_text(
            f"# Stage 2 {args.variant} Verifier\n\n"
            "Status: skipped.\n\n"
            f"Reason: {status['reason']}\n",
            encoding="utf-8",
        )
        return 0 if args.allow_skip else 2
    rows = read_jsonl(feature_manifest)
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    test_rows = [row for row in rows if row["split"] == "test"]
    if not train_rows:
        write_json(out_dir / "metrics.json", {"status": "blocked", "reason": "no_training_rows"})
        return 2
    sample = np.load(train_rows[0]["feature_path"])
    sample_arr = sample["features"] if "features" in sample else sample[sample.files[0]]
    input_dim = int(sample_arr.shape[-1])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if config.get("stage2", {}).get("allow_cpu_qwen", True) is False and device != "cuda:0":
        write_json(out_dir / "metrics.json", {"status": "blocked", "reason": "strict Stage 2 requires CUDA"})
        return 2
    model = QwenVerifierHead(
        input_dim,
        projection_dim=int(config["stage2"]["projection_dim"]),
        embedding_dim=int(config["stage2"]["embedding_dim"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["stage2"]["learning_rate"]),
        weight_decay=float(config["stage2"]["weight_decay"]),
    )
    pos = sum(row["label"] == 1 for row in train_rows)
    neg = sum(row["label"] == 0 for row in train_rows)
    pos_weight = torch.tensor([min(10.0, max(0.1, neg / pos))], dtype=torch.float32, device=device) if pos and neg else None
    loader = DataLoader(
        QwenFeatureDataset(train_rows),
        batch_size=int(config["stage2"]["batch_size"]),
        shuffle=True,
        collate_fn=collate,
    )
    lambda_supcon = float(config["stage2"]["lambda_supcon"]) if args.variant == "bce_supcon" else 0.0
    history = []
    best_state = None
    best_val = -1.0
    patience = 0
    no_pair_batches = 0
    for epoch in range(1, int(config["stage2"]["epochs"]) + 1):
        model.train()
        losses = []
        for hidden, mask, labels, phrase_ids, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            result = model(hidden.to(device), mask.to(device))
            bce = bce_with_logits_loss(result["logit"], labels.to(device), pos_weight=pos_weight)
            supcon = supervised_contrastive_loss(
                result["embedding"],
                phrase_ids,
                temperature=float(config["stage2"]["temperature"]),
            )
            if lambda_supcon and float(supcon.detach().cpu().item()) == 0.0:
                no_pair_batches += 1
            loss = bce + lambda_supcon * supcon
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["stage2"]["gradient_clip_norm"]))
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        val_pred = predict(model, val_rows, device)
        if val_pred:
            val_metrics = binary_metrics([p["label"] for p in val_pred], [p["stage2_score"] for p in val_pred], 0.5)
            val_score = val_metrics.get("f1") if val_metrics.get("f1") is not None else 0.0
        else:
            val_metrics = {"n": 0}
            val_score = -float(np.mean(losses))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_score": float(val_score), "val_metrics": val_metrics})
        if val_score > best_val:
            best_val = float(val_score)
            best_state = {k: value.detach().cpu() for k, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= int(config["stage2"]["early_stopping_patience"]):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    val_pred = predict(model, val_rows, device)
    test_pred = predict(model, test_rows, device)
    threshold = select_recall_first_threshold(
        [p["label"] for p in val_pred],
        [p["stage2_score"] for p in val_pred],
        float(config["stage2"]["recall_target"]),
    )
    theta = float(threshold["threshold"])
    qwen_integrity_path = Path(args.run_dir) / "stage2_qwen_features" / "frozen_qwen_integrity.json"
    qwen_integrity = read_json(qwen_integrity_path) if qwen_integrity_path.exists() else {"status": "missing"}
    head_integrity = checksum_representative_parameters(model)
    metrics = {
        "status": "ok",
        "variant": args.variant,
        "device": device,
        "input_feature_dim": input_dim,
        "lambda_supcon": lambda_supcon,
        "no_pair_batches": no_pair_batches,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "test_rows": len(test_rows),
        "theta_2": theta,
        "threshold_selection": threshold,
        "test_metrics": binary_metrics([p["label"] for p in test_pred], [p["stage2_score"] for p in test_pred], theta),
        "verifier_parameters": head_integrity.total_parameters,
        "verifier_trainable_parameters": head_integrity.trainable_parameters,
        "qwen_integrity": qwen_integrity,
    }
    torch.save({"model_state": model.state_dict(), "input_dim": input_dim, "config": config["stage2"]}, out_dir / "checkpoint_best.pt")
    write_json(out_dir / "threshold.json", threshold)
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "frozen_qwen_integrity.json", qwen_integrity)
    write_jsonl(out_dir / "val_predictions.jsonl", val_pred)
    write_jsonl(out_dir / "test_predictions.jsonl", test_pred)
    with (out_dir / "train_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_score"])
        writer.writeheader()
        for row in history:
            writer.writerow({k: row[k] for k in ("epoch", "train_loss", "val_score")})
    (out_dir / "report.md").write_text(
        f"# Stage 2 {args.variant} Verifier\n\n"
        "- Status: ok\n"
        f"- Device: `{device}`\n"
        f"- Theta 2: {theta:.6f}\n"
        f"- Test metrics: `{metrics['test_metrics']}`\n"
        f"- Qwen integrity status: `{qwen_integrity.get('status')}`\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
