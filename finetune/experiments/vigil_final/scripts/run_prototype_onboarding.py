#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np

from vigil_final.prototype import PrototypeRecipe, apply_recipe, build_prototype, cosine_similarity, validate_support_rows
from vigil_final.utils import read_json, read_jsonl, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-index", required=True)
    parser.add_argument("--support-json", required=True)
    parser.add_argument("--recipe-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.embedding_index)
    support_ids = set(read_json(args.support_json)["support_clip_ids"])
    support = [row for row in rows if str(row["clip_id"]) in support_ids]
    query = [row for row in rows if str(row["clip_id"]) not in support_ids]
    validate_support_rows(support, query, shots=len(support_ids))
    proto = build_prototype([np.asarray(row["embedding"], dtype=np.float32) for row in support])
    recipe_data = read_json(args.recipe_json)
    recipe = PrototypeRecipe(
        method=recipe_data.get("method", "base_plus_prototype"),
        alpha=float(recipe_data.get("alpha", 0.5)),
        beta=float(recipe_data.get("beta", 0.0)),
        threshold=float(recipe_data.get("threshold", 0.5)),
        top_k=int(recipe_data.get("top_k", 3)),
    )
    predictions = []
    for row in query:
        sim = cosine_similarity(np.asarray(row["embedding"], dtype=np.float32), proto)
        score, decision = apply_recipe(float(row.get("base_stage2_logit", 0.0)), sim, recipe)
        predictions.append({"clip_id": row["clip_id"], "label": row["label"], "prototype_similarity": sim, "score": score, "decision": decision})
    write_json(args.output, {"status": "ok", "support": len(support), "query": len(query), "predictions": predictions})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
