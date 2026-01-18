#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8080}"

if [[ -f "venv/bin/activate" ]]; then
  source venv/bin/activate
fi

curl -s -X POST "${BASE_URL}/cache/clear" >/dev/null

run_query() {
  local q="$1"
  local payload
  payload=$(Q="$q" python - <<'PY'
import json
import os
print(json.dumps({"query": os.environ["Q"]}))
PY
)
  local body_file
  body_file="$(mktemp)"
  local status
  status=$(curl -s -o "${body_file}" -w "%{http_code}" -X POST "${BASE_URL}/query" \
    -H "Content-Type: application/json" \
    -d "${payload}")
  if [[ -z "${status}" ]]; then
    echo "cached=error latency_ms=error"
    rm -f "${body_file}"
    return
  fi
  if [[ "${status}" != "200" ]]; then
    echo "cached=error latency_ms=error"
    rm -f "${body_file}"
    return
  fi
  python - "${body_file}" <<'PY'
import json
import sys
try:
    with open(sys.argv[1], "r") as fh:
        data = json.load(fh)
    print(f"cached={data.get('cached')} latency_ms={data.get('latency_ms')}")
except Exception:
    print("cached=error latency_ms=error")
PY
  rm -f "${body_file}"
}

run_query "How do I reset my password?"
run_query "How do I reset my password?"
run_query "What is the process for resetting my password?"
