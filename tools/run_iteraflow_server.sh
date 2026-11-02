#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${ITERAFLOW_INPUTS:?Set ITERAFLOW_INPUTS to a JSON map of server dataset paths}"
: "${ITERAFLOW_OUTPUT_ROOT:?Set ITERAFLOW_OUTPUT_ROOT to a writable experiment directory}"
args=(--inputs "$ITERAFLOW_INPUTS" --output-root "$ITERAFLOW_OUTPUT_ROOT")
if [[ -n "${ITERAFLOW_R4_OVERRIDES:-}" ]]; then
  args+=(--r4-overrides "$ITERAFLOW_R4_OVERRIDES")
fi
if [[ -n "${ITERAFLOW_CUSTOM_OPERATOR_PATH:-}" ]]; then
  args+=(--custom-operator-path "$ITERAFLOW_CUSTOM_OPERATOR_PATH")
fi
if [[ -n "${ITERAFLOW_WORKLOAD:-}" ]]; then
  args+=(--workload "$ITERAFLOW_WORKLOAD")
fi
if [[ -n "${ITERAFLOW_SYSTEM:-}" ]]; then
  args+=(--system "$ITERAFLOW_SYSTEM")
fi
if [[ -n "${ITERAFLOW_ROUND:-}" ]]; then
  args+=(--round "$ITERAFLOW_ROUND")
fi
if [[ "${ITERAFLOW_RUN:-0}" == "1" ]]; then
  args+=(--run)
fi
if [[ "${ITERAFLOW_RERUN:-0}" == "1" ]]; then
  args+=(--rerun)
fi
cd "$repo_root"
exec python tools/iteraflow_server_matrix.py "${args[@]}"
