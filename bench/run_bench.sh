#!/usr/bin/env bash
set -euo pipefail

DUR=30
MODE="server"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --duration) DUR="$2"; shift 2;;
    --mode) MODE="$2"; shift 2;;
    *) shift;;
  esac
done

export MODE
echo "Ensure the page is open on laptop+phone and streaming."
echo "Collecting metrics for ${DUR}s in MODE=${MODE} ..."

# trigger a fresh metrics window (optional)
curl -s -X POST http://localhost:8000/metrics/reset >/dev/null || true

sleep "${DUR}"

curl -s http://localhost:8000/metrics > ./metrics.json
echo "Wrote bench/metrics.json"
