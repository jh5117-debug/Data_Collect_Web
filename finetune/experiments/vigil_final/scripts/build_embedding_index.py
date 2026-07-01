#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from vigil_final.utils import read_json, read_jsonl, write_json, write_jsonl
from vigil_two_stage.stage2_model import QwenVerifierHead


def load_feature(path: str) -> np.ndarray:
    data = np.load(path)
    return (data["features"] if "features" in data else data[data.files[0]]).astype(np.float32)


def embed_rows(checkpoint_path: Path, rows: list[dict], device: str) -> list[dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = QwenVerifierHead(int(ckpt["input_dim"]), projection_dim=int(ckpt["config"]["projection_dim"]), embedding_dim=int(ckpt["config"]["embedding_dim"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    out = []
    checksum = checkpoint_path.stat().st_size
    with torch.no_grad():
        for row in rows:
            arr = load_feature(row["feature_path"])
            hidden = torch.from_numpy(arr).unsqueeze(0).to(device)
            mask = torch.ones(1, arr.shape[0], dtype=torch.bool, device=device)
            result = model(hidden, mask)
            embedding = result["embedding"].cpu().numpy()[0].astype(float).tolist()
            norm = float(np.linalg.norm(np.asarray(embedding)))
            out.append(
                {
                    "participant_alias": row["participant_alias"],
                    "fold": row.get("fold"),
                    "clip_id": row["clip_id"],
                    "window_index": row.get("window_index", 0),
                    "label": row["label"],
                    "prompt_group": row.get("prompt_group"),
                    "phrase_id": row.get("phrase_id"),
                    "embedding": embedding,
                    "embedding_l2_norm": norm,
                    "base_stage2_logit": float(result["logit"].cpu().item()),
                    "source_checkpoint_size_checksum": checksum,
                }
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--variant", default="stage2_bce")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    rows = read_jsonl(run_dir / "stage2_qwen_features" / "qwen_features_manifest.jsonl")
    checkpoint = run_dir / args.variant / "checkpoint_best.pt"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    embedded = embed_rows(checkpoint, rows, device)
    bad = [row for row in embedded if not np.isfinite(np.asarray(row["embedding"])).all() or abs(float(row["embedding_l2_norm"]) - 1.0) > 1e-4]
    if bad:
        raise RuntimeError(f"bad embeddings: {len(bad)}")
    write_jsonl(args.output, embedded)
    write_json(str(Path(args.output).with_suffix(".summary.json")), {"status": "ok", "rows": len(embedded), "checkpoint": str(checkpoint)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
