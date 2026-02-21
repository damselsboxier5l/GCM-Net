import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO11/YOLO26 with Ultralytics (FADE).")
    parser.add_argument(
        "--model",
        type=str,
        default="ultralytics/cfg/models/26/yolo26-fade.yaml",
        help="Model name or path, e.g. yolo11n.pt, yolo26s.pt, or local yaml.",
    )
    parser.add_argument("--data", type=str, default="data/paowu6.yaml", help="Dataset yaml path")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0", help="cuda device id, e.g. 0 or 0,1 or cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=str, default="runs/ultra_train")
    parser.add_argument("--name", type=str, default="exp")
    parser.add_argument("--optimizer", type=str, default="auto")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--amp", action="store_true", help="Enable Automatic Mixed Precision. Disabled by default for DCN stability.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--cache",
        nargs="?",
        const="ram",
        default=False,
        choices=("ram", "disk"),
        help="Cache images for faster training. Use '--cache' (ram) or '--cache disk'.",
    )
    parser.add_argument("--cos-lr", action="store_true")
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument(
        "--local-ultralytics",
        type=str,
        default=".",
        help="Local repo path to prioritize over pip package.",
    )
    return parser.parse_args()


def normalize_model_name(model: str) -> str:
    known_prefixes = ("yolo11", "yolo26")
    path_model = Path(model)
    if path_model.exists():
        return model
    if model.endswith((".pt", ".yaml", ".yml")):
        return model
    if model.startswith(known_prefixes):
        return f"{model}.pt"
    return model


def main():
    args = parse_args()
    local_ultra = Path(args.local_ultralytics)
    if local_ultra.exists():
        sys.path.insert(0, str(local_ultra.resolve()))
    model_name = normalize_model_name(args.model)

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("Failed to import ultralytics. Install dependencies with: pip install -r requirements.txt") from exc

    model = YOLO(model_name)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        optimizer=args.optimizer,
        lr0=args.lr0,
        patience=args.patience,
        amp=args.amp,
        resume=args.resume,
        cache=args.cache,
        cos_lr=args.cos_lr,
        close_mosaic=args.close_mosaic,
    )


if __name__ == "__main__":
    main()
