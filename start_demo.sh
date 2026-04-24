#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"

if [ ! -d "$repo_dir/frontend/node_modules" ]; then
    echo "[FilingDelta] Frontend dependencies are missing."
    echo "Please run: cd frontend && npm install"
    exit 1
fi

log_dir="$repo_dir/data/outputs/logs"
mkdir -p "$log_dir"

backend_log="$log_dir/backend.log"
frontend_log="$log_dir/frontend.log"
backend_url="http://127.0.0.1:8000"
backend_health_url="$backend_url/health"
frontend_url="http://127.0.0.1:5173"
backend_pid=""
frontend_pid=""

cleanup() {
    if [ -n "$frontend_pid" ] && kill -0 "$frontend_pid" 2>/dev/null; then
        kill "$frontend_pid" 2>/dev/null || true
    fi
    if [ -n "$backend_pid" ] && kill -0 "$backend_pid" 2>/dev/null; then
        kill "$backend_pid" 2>/dev/null || true
    fi
}

wait_for_http() {
    local url="$1"
    local attempts="${2:-30}"
    local attempt

    if ! command -v curl >/dev/null 2>&1; then
        sleep 2
        return 0
    fi

    for ((attempt = 1; attempt <= attempts; attempt++)); do
        if curl -fsS "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done

    return 1
}

open_frontend() {
    if [ "$(uname -s)" = "Darwin" ] && command -v open >/dev/null 2>&1; then
        open "$frontend_url" >/dev/null 2>&1 || true
    fi
}

trap cleanup INT TERM EXIT

echo "[FilingDelta] Starting backend..."
"$repo_dir/start_backend.sh" >"$backend_log" 2>&1 &
backend_pid=$!

if ! kill -0 "$backend_pid" 2>/dev/null; then
    echo "[FilingDelta] Backend failed to start. Last log lines:"
    tail -n 40 "$backend_log" || true
    exit 1
fi

if ! wait_for_http "$backend_health_url" 30; then
    echo "[FilingDelta] Backend did not become ready at $backend_health_url. Last log lines:"
    tail -n 40 "$backend_log" || true
    exit 1
fi

echo "[FilingDelta] Starting frontend..."
"$repo_dir/start_frontend.sh" >"$frontend_log" 2>&1 &
frontend_pid=$!

if ! kill -0 "$frontend_pid" 2>/dev/null; then
    echo "[FilingDelta] Frontend failed to start. Last log lines:"
    tail -n 40 "$frontend_log" || true
    exit 1
fi

if ! wait_for_http "$frontend_url" 30; then
    echo "[FilingDelta] Frontend did not become ready at $frontend_url. Last log lines:"
    tail -n 40 "$frontend_log" || true
    exit 1
fi

open_frontend

echo "[FilingDelta] Demo is starting."
echo "  Backend:  $backend_url"
echo "  Frontend: $frontend_url"
echo "  Backend log:  $backend_log"
echo "  Frontend log: $frontend_log"
echo "  macOS double-click entry: start_demo.command"
echo "[FilingDelta] Press Ctrl-C to stop both processes."

while true; do
    if ! kill -0 "$backend_pid" 2>/dev/null; then
        echo "[FilingDelta] Backend stopped. Last log lines:"
        tail -n 40 "$backend_log" || true
        exit 1
    fi
    if ! kill -0 "$frontend_pid" 2>/dev/null; then
        echo "[FilingDelta] Frontend stopped. Last log lines:"
        tail -n 40 "$frontend_log" || true
        exit 1
    fi
    sleep 2
done
