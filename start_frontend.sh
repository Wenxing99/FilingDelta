#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir/frontend"

if [ ! -d "node_modules" ]; then
    echo "[FilingDelta] Frontend dependencies are missing."
    echo "Please run: cd frontend && npm install"
    exit 1
fi

exec npm run dev -- --host 127.0.0.1 --port 5173
