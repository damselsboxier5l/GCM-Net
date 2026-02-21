import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Predict with YOLO11/YOLO26 via Ultralytics.")
    parser.add_argument(
        "--model",
        type=str,
        default="ultralytics/cfg/models/26/yolo26-fade.yaml",
        help="Model name or path, e.g. yolo11n.pt, yolo26s.pt, or local yaml.",
    )
    parser.add_argument("--source", type=str, default="data/images", help="Image/video/file/dir/stream source")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--project", type=str, default="runs/ultra_predict")
    parser.add_argument("--name", type=str, default="exp")
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--save-conf", action="store_true")
    parser.add_argument("--save-crop", action="store_true")
    parser.add_argument("--vid-stride", type=int, default=1)
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
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        max_det=args.max_det,
        project=args.project,
        name=args.name,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        save_crop=args.save_crop,
        vid_stride=args.vid_stride,
    )


if __name__ == "__main__":
    main()
