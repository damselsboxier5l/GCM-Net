#!/usr/bin/env bash
set -euo pipefail

MODEL="ultralytics/cfg/models/26/yolo26-fade.yaml"
SOURCE="data/images"
DEVICE="0"

pdm run python -u predict.py \
  --model "${MODEL}" \
  --source "${SOURCE}" \
  --device "${DEVICE}" \
  "$@"
