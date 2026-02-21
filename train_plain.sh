#!/usr/bin/env bash
set -euo pipefail

# Pure YOLO11/YOLO26 training on paowu (3-channel), no FADE/MASS config.
# Usage:
#   bash train_plain.sh yolo26n.pt
#   bash train_plain.sh yolo11n.pt
#   bash train_plain.sh yolo26s.pt --epochs 200 --device 0

MODEL="${1:-yolo26n.pt}"
if [[ $# -gt 0 ]]; then
  shift
fi

DATA="data/paowu.yaml"
IMGSZ=960
BATCH=8
EPOCHS=100
DEVICE="6"
WORKERS=8
PROJECT="runs/ultra_train"
NAME="plain-${MODEL%.*}"
LOG_DIR="logs"
RESUME_CKPT="${RESUME_CKPT:-}"
USE_AMP="${USE_AMP:-1}"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_plain_${MODEL%.*}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[train_plain.sh] model=${MODEL}"
echo "[train_plain.sh] logging to ${LOG_FILE}"

CMD=(
  pdm run python -u train.py
  --data "${DATA}"
  --imgsz "${IMGSZ}"
  --batch "${BATCH}"
  --epochs "${EPOCHS}"
  --device "${DEVICE}"
  --workers "${WORKERS}"
#   --cache disk
  --project "${PROJECT}"
  --name "${NAME}"
)

if [[ -n "${RESUME_CKPT}" ]]; then
  CMD+=(--model "${RESUME_CKPT}" --resume)
  echo "[train_plain.sh] resume from ${RESUME_CKPT}"
else
  CMD+=(--model "${MODEL}")
fi

if [[ "${USE_AMP}" == "1" ]]; then
  CMD+=(--amp)
  echo "[train_plain.sh] amp enabled"
fi

CMD+=("$@")
"${CMD[@]}"
