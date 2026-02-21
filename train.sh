#!/usr/bin/env bash
set -euo pipefail

MODEL="ultralytics/cfg/models/26/yolo26l-fade.yaml"
DATA="data/paowu6.yaml"
IMGSZ=960
BATCH=16
EPOCHS=100
DEVICE="4,5"
WORKERS=8
PROJECT="runs/ultra_train"
NAME="mass-train"
LOG_DIR="logs"
RESUME_CKPT="${RESUME_CKPT:-}"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[train.sh] logging to ${LOG_FILE}"

export CUDA_VISIBLE_DEVICES="${DEVICE}"

CMD=(
  pdm run python -u train.py
  --data "${DATA}"
  --imgsz "${IMGSZ}"
  --batch "${BATCH}"
  --epochs "${EPOCHS}"
  --device "${DEVICE}"
  --workers "${WORKERS}"
  --cache "disk"
  --project "${PROJECT}"
  --name "${NAME}"
)

if [[ -n "${RESUME_CKPT}" ]]; then
  CMD+=(--model "${RESUME_CKPT}" --resume)
  echo "[train.sh] resume from ${RESUME_CKPT}"
else
  CMD+=(--model "${MODEL}")
fi

CMD+=("$@")
"${CMD[@]}"
