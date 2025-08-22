#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-wasm}"  # wasm | server
if [[ "${1:-}" == "--mode" && -n "${2:-}" ]]; then
  MODE="$2"
fi

export MODE
echo "Starting in MODE=${MODE}"
docker-compose up --build -d

echo ""
echo "Open http://localhost:3000 on your laptop."
echo "On your phone, open the same URL (same Wi-Fi) or run: $0 --ngrok"
if [[ "${1:-}" == "--ngrok" ]]; then
  echo "------ NGROK QUICK TIP ------"
  echo "Install ngrok and run: ngrok http 3000"
  echo "Share the printed https URL with your phone."
fi
