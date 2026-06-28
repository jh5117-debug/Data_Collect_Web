#!/usr/bin/env bash
set -euo pipefail

BASE="${1:-http://127.0.0.1:7861}"
curl -fsS "${BASE}/health"
echo
