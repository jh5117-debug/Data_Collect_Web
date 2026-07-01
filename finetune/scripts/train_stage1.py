#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from vigil_two_stage.losses import bce_with_logits_loss
from vigil_two_stage.metrics import binary_metrics
from vigil_two_stage.stage1_model import Stage1GRUClassifier, count_parameters
from vigil_two_stage.thresholds import select_recall_first_threshold
from vigil_two_stage.utils import ensure_dir, read_jsonl, seed_everything, write_json, write_jsonl


class FeatureDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        arr = np.load(row["feature_path"])["features"].astype(np.float32)
        return arr, int(row["label"]), row


def collate(batch):
    arrays, labels, rows = zip(*batch)
    lengths = torch.tensor([a.shape[0] for a in arrays], dtype=torch.long)
    max_len = int(lengths.max().item())
    dim = arrays[0].shape[1]
    x = torch.zeros(len(arrays), max_len, dim, dtype=torch.float32)
    for i, arr in enumerate(arrays):
        x[i, : arr.shape[0]] = torch.from_numpy(arr)
    y = torch.tensor(labels, dtype=torch.float32)
    return x, lengths, y, rows


def predict(model, rows, device):
    if not rows:
        return []
    loader = DataLoader(FeatureDataset(rows), batch_size=32, shuffle=False, collate_fn=collate)
    out = []
    model.eval()
    with torch.no_grad():
        for x, lengths, y, batch_rows in loader:
            scores = torch.sigmoid(model(x.to(device), lengths.to(device))).cpu().numpy().tolist()
            for row, score in zip(batch_rows, scores):
                pred = {k: row[k] for k in ("clip_id", "speaker_id", "session_id", "prompt_group", "transcript", "label", "phrase_id", "split")}
                if "window_index" in row:
                    pred["window_index"] = row["window_index"]
                pred["score"] = float(score)
                out.append(pred)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed_everything(int(config["seed"]))
    run_dir = ensure_dir(Path(args.run_dir) / "stage1")
    rows = read_jsonl(args.features_manifest)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]
    if not train_rows:
        write_json(run_dir / "metrics.json", {"status": "blocked", "reason": "no_training_rows"})
        return 2
    input_dim = int(rows[0]["feature_dim"])
    model = Stage1GRUClassifier(
        input_dim,
        hidden_size=int(config["stage1"]["gru_hidden_size"]),
        layers=int(config["stage1"]["gru_layers"]),
        dropout=float(config["stage1"]["dropout"]),
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    pos = sum(r["label"] == 1 for r in train_rows)
    neg = sum(r["label"] == 0 for r in train_rows)
    pos_weight = None
    if pos and neg:
        pos_weight = torch.tensor([min(10.0, max(0.1, neg / pos))], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["stage1"]["learning_rate"]), weight_decay=float(config["stage1"]["weight_decay"]))
    loader = DataLoader(
        FeatureDataset(train_rows),
        batch_size=int(config["stage1"]["batch_size"]),
        shuffle=True,
        collate_fn=collate,
    )
    history = []
    best_val = -1.0
    best_state = None
    patience = 0
    for epoch in range(1, int(config["stage1"]["epochs"]) + 1):
        model.train()
        losses = []
        for x, lengths, y, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x.to(device), lengths.to(device))
            loss = bce_with_logits_loss(logits, y.to(device), pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["stage1"]["gradient_clip_norm"]))
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        val_pred = predict(model, val_rows, device)
        if val_pred:
            threshold_tmp = 0.5
            val_metrics = binary_metrics([p["label"] for p in val_pred], [p["score"] for p in val_pred], threshold_tmp)
            val_score = val_metrics.get("f1") if val_metrics.get("f1") is not None else 0.0
        else:
            val_metrics = {"n": 0}
            val_score = -float(np.mean(losses))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_score": float(val_score), "val_metrics": val_metrics})
        if val_score > best_val:
            best_val = float(val_score)
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= int(config["stage1"]["early_stopping_patience"]):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    val_pred = predict(model, val_rows, device)
    test_pred = predict(model, test_rows, device)
    threshold = select_recall_first_threshold(
        [p["label"] for p in val_pred],
        [p["score"] for p in val_pred],
        float(config["stage1"]["recall_target"]),
    )
    theta = float(threshold["threshold"])
    metrics = {
        "status": "ok",
        "device": device,
        "feature_backend": rows[0].get("feature_backend"),
        "official_openwakeword_used": rows[0].get("feature_backend") == "official_openwakeword",
        "input_feature_dim": input_dim,
        "parameters": count_parameters(model),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "test_rows": len(test_rows),
        "pos_weight": float(pos_weight.item()) if pos_weight is not None else None,
        "theta_1": theta,
        "threshold_selection": threshold,
        "test_metrics": binary_metrics([p["label"] for p in test_pred], [p["score"] for p in test_pred], theta),
    }
    torch.save({"model_state": model.state_dict(), "input_dim": input_dim, "config": config["stage1"]}, run_dir / "checkpoint_best.pt")
    write_json(run_dir / "model_config.json", {"input_dim": input_dim, **config["stage1"]})
    write_json(run_dir / "threshold.json", threshold)
    write_json(run_dir / "metrics.json", metrics)
    write_jsonl(run_dir / "val_predictions.jsonl", val_pred)
    write_jsonl(run_dir / "test_predictions.jsonl", test_pred)
    with (run_dir / "train_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_score"])
        writer.writeheader()
        for row in history:
            writer.writerow({k: row[k] for k in ["epoch", "train_loss", "val_score"]})
    (run_dir / "report.md").write_text(
        "# Stage 1 Smoke Result\n\n"
        f"- Status: ok\n"
        f"- Feature backend: `{metrics['feature_backend']}`\n"
        f"- Official openWakeWord used: {metrics['official_openwakeword_used']}\n"
        f"- Theta 1: {theta:.6f}\n"
        f"- Test metrics: `{metrics['test_metrics']}`\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
