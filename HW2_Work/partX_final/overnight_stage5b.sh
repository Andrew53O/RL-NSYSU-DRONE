#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

BASE_MODEL="models/stage5/variantA/run005/best/best_average_model.zip"

if [[ ! -f "$BASE_MODEL" ]]; then
  echo "Missing resume model: $BASE_MODEL"
  exit 1
fi

echo "=== Stage 5B train from $BASE_MODEL ==="
python3 train.py \
  --stage 5 \
  --variant B \
  --resume-from "$BASE_MODEL" \
  --timesteps 60000 \
  --success-distance 0.10 \
  --max-steps 1000 \
  --step-dt 0.05 \
  --learning-rate 3e-4 \
  --n-steps 512 \
  --batch-size 64 \
  --gamma 0.99 \
  --checkpoint-freq 10000 \
  --best-window 20 \
  --plateau-window 50 \
  --plateau-patience 50 \
  --plateau-min-delta 1.0 \
  --near-target-action-penalty 0.3 \
  --action-penalty 0.03 \
  --action-smoothness-penalty 0.09 \
  --log-position-every 25

LATEST_MODEL="$(ls -td models/stage5/variantB/run*/best/best_average_model.zip | head -n 1)"

if [[ ! -f "$LATEST_MODEL" ]]; then
  echo "Could not find latest Stage 5B best model."
  exit 1
fi

echo "=== Stage 5B test with $LATEST_MODEL ==="
python3 test.py \
  --stage 5 \
  --variant B \
  --model "$LATEST_MODEL" \
  --success-distance 0.10 \
  --episodes 10 \
  --max-steps 1000 \
  --step-dt 0.05 \
  --log-position-every 25
