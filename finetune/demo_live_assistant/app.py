from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from model_runtime import DEFAULT_RUN_DIR
from server import create_app


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    app = create_app(run_dir=Path(args.run_dir), force_mock=bool(args.mock), load_models=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
