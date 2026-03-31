#!/bin/bash
#SBATCH --gpus=a100l:2
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH --partition=long
#SBATCH --output=%x.out
#SBATCH --error=%x.out

set -euo pipefail

CONFIG_NAME="${1:-tas_rps_naive_vs_gptoss}"

export OPENAI_API_KEY=EMPTY
export VLLM_HOST=127.0.0.1
VLLM_PORT="$(
  python - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
port = s.getsockname()[1]
s.close()
print(port)
PY
)"
export VLLM_PORT=8000
export VLLM_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM
# export TORCHDYNAMO_DISABLE=1
CUDA_VISIBLE_DEVICES=0 vllm serve openai/gpt-oss-120b \
  --host ${VLLM_HOST} --port ${VLLM_PORT} \
  --max-model-len 10_000 \
  --async-scheduling \
  --no-enable-prefix-caching \
  > vllm-${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!

python - <<'PY'
import time, requests, os
base = os.environ["VLLM_BASE_URL"]
for _ in range(300):
    try:
        r = requests.get(base + "/models", timeout=2)
        if r.status_code == 200:
            raise SystemExit(0)
    except Exception:
        pass
    time.sleep(1)
raise SystemExit("vLLM not ready")
PY

python - <<'PY' &
import time, os, requests, signal
base = os.environ["VLLM_BASE_URL"]
while True:
    try:
        r = requests.get(base + "/models", timeout=2)
        if r.status_code != 200:
            raise RuntimeError
    except Exception:
        os.kill(os.getpid(), signal.SIGTERM)
    time.sleep(5)
PY
WATCHDOG_PID=$!

CUDA_VISIBLE_DEVICES=1 python run.py --config-name="${CONFIG_NAME}" &
TRAIN_PID=$!

wait "${TRAIN_PID}"
kill "${WATCHDOG_PID}" 2>/dev/null || true
wait "${WATCHDOG_PID}" 2>/dev/null || true
