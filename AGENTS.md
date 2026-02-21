# Repository Guidelines

## Project Structure & Module Organization
- Main pipeline (YOLO26/YOLO11): `ultralytics/`, `train.py`, `predict.py`, `data/`.
- FADE custom configs:
  - Models: `ultralytics/cfg/models/11/yolo11-fade.yaml`, `ultralytics/cfg/models/26/yolo26-fade.yaml`
  - Dataset: `data/paowu6.yaml` (`channels: 6`, `paired_mask: true`).
- Data assets: `dataset/` (train/val files), `assets/` (docs/media), `result/` (evaluation outputs).

## Build, Test, and Development Commands
- Install deps: `pdm install`
- Train (YOLO26 FADE):  
  `pdm run python -u train.py --model ultralytics/cfg/models/26/yolo26-fade.yaml --data data/paowu6.yaml --imgsz 960 --device 0`
- Predict:  
  `pdm run python -u predict.py --model ultralytics/cfg/models/26/yolo26-fade.yaml --source data/images`
- Quick syntax check:  
  `python -m py_compile train.py predict.py`

## Coding Style & Naming Conventions
- Python: 4-space indentation, PEP 8 style, descriptive names, avoid one-letter variables.
- Keep changes minimal and scoped; do not refactor unrelated modules.
- New configs should follow existing naming: `yoloXX-fade.yaml`, dataset config `*-6.yaml` for 6-channel variants.

## Testing Guidelines
- No enforced unit-test suite in this repo; validate by:
  1) `py_compile` for touched Python files,  
  2) one-epoch smoke run (`--epochs 1 --workers 0 --batch 2`),  
  3) confirm train loop starts (not just model summary).
- For server debugging, always capture full logs and exit code.

## Commit & Pull Request Guidelines
- This working tree may not include full Git history; use clear Conventional Commit style:
  - `feat:`, `fix:`, `refactor:`, `docs:`.
- PRs should include:
  - Problem statement and scope,
  - Files changed and why,
  - Repro command(s),
  - Key log snippets (startup + first train iteration),
  - Any environment assumptions (CUDA, torch/torchvision versions).

## Security & Configuration Tips
- Do not hardcode private paths, credentials, or tokens.
- Prefer relative paths in configs.
- `DCN` stability note: enable torchvision DCN only when verified (`FADE_ENABLE_TV_DCN=1`).
