#!/usr/bin/env python
"""Simplified YOLOv8 dataset pipeline."""
import argparse
import random
import shutil
from pathlib import Path
from ultralytics import YOLO
import torch
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

DEFAULT_DEVICE = "0" if torch.cuda.is_available() else "cpu"

def find_image(label: Path, images_dir: Path) -> Path | None:
    for ext in IMAGE_EXTS:
        img = images_dir / f"{label.stem}{ext}"
        if img.exists():
            return img
    return None


def prepare_dataset(
    images: Path,
    labels: Path,
    out: Path,
    train_ratio: float,
    val_ratio: float,
    move: bool,
) -> None:
    random.seed(42)
    train_img = out / "images" / "train"
    val_img = out / "images" / "val"
    test_img = out / "images" / "test"
    train_lbl = out / "labels" / "train"
    val_lbl = out / "labels" / "val"
    test_lbl = out / "labels" / "test"
    for d in (train_img, val_img, test_img, train_lbl, val_lbl, test_lbl):
        d.mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[Path, Path]] = []
    for lbl in labels.glob("*.txt"):
        img = find_image(lbl, images)
        if img is None:
            print(f"[WARN] image for label '{lbl.name}' not found – skipping")
            continue
        pairs.append((img, lbl))

    if not pairs:
        print("No image/label pairs found – nothing to do")
        return

    random.shuffle(pairs)
    split1 = int(len(pairs) * train_ratio)
    split2 = int(len(pairs) * (train_ratio + val_ratio))
    train_pairs = pairs[:split1]
    val_pairs = pairs[split1:split2]
    test_pairs = pairs[split2:]

    def transfer(src: Path, dst: Path):
        if move:
            shutil.move(src, dst)
        else:
            shutil.copy2(src, dst)

    for img, lbl in train_pairs:
        transfer(img, train_img / img.name)
        transfer(lbl, train_lbl / lbl.name)

    for img, lbl in val_pairs:
        transfer(img, val_img / img.name)
        transfer(lbl, val_lbl / lbl.name)

    for img, lbl in test_pairs:
        transfer(img, test_img / img.name)
        transfer(lbl, test_lbl / lbl.name)

    print(
        f"Done! Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}"
    )


def train_model(data: Path, model: str, epochs: int, imgsz: int, batch: int, device: str) -> None:
    yolo = YOLO(model)
    yolo.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="yolo/runs",
    )


def latest_weights() -> Path | None:
    root = Path("yolo/runs/detect")
    if not root.exists():
        return None
    dirs = sorted(
        (p for p in root.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for d in dirs:
        for name in ("best.pt", "last.pt"):
            w = d / "weights" / name
            if w.exists():
                return w
    return None


def predict_image(source: Path, weights: Path | None, imgsz: int, device: str) -> None:
    if weights is None:
        weights = latest_weights()
        if weights is None:
            raise SystemExit("No trained weights found in yolo/runs")
    yolo = YOLO(str(weights))
    results = yolo.predict(
        str(source), imgsz=imgsz, device=device, save=True, project="yolo/runs"
    )
    if results:
        print(f"Results saved to {results[0].save_dir}")


def test_model(data: Path, weights: Path | None, imgsz: int, device: str) -> None:
    if weights is None:
        weights = latest_weights()
        if weights is None:
            raise SystemExit("No trained weights found in yolo/runs")
    yolo = YOLO(str(weights))
    metrics = yolo.val(
        data=str(data),
        split="test",
        imgsz=imgsz,
        device=device,
        save=True,
        project="yolo/runs",
    )
    if hasattr(metrics, "save_dir"):
        print(f"Results saved to {metrics.save_dir}")
    print("mAP50-95:", metrics.box.map)
    print("mAP50:", metrics.box.map50)


def export_cards(source: Path, out: Path, weights: Path | None, imgsz: int, device: str) -> None:
    if weights is None:
        weights = latest_weights()
        if weights is None:
            raise SystemExit("No trained weights found in yolo/runs")
    yolo = YOLO(str(weights))
    imgs = []
    if source.is_dir():
        for ext in IMAGE_EXTS:
            imgs.extend(sorted(source.glob(f"*{ext}")))
    else:
        imgs.append(source)
    out.mkdir(parents=True, exist_ok=True)
    for img_path in imgs:
        results = yolo.predict(str(img_path), imgsz=imgsz, device=device)
        if not results:
            continue
        res = results[0]
        im = Image.open(img_path)
        boxes = res.boxes.xyxy.cpu().numpy().astype(int)
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            crop = im.crop((x1, y1, x2, y2))
            dest = out / f"{img_path.stem}_{i}{img_path.suffix}"
            crop.save(dest)
            print(f"Saved {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8 pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare", help="Prepare dataset structure")
    p_prep.add_argument(
        "--images",
        default="yolo/images",
        help="Source images directory",
    )
    p_prep.add_argument(
        "--labels",
        default="yolo/labels",
        help="Source labels directory",
    )
    p_prep.add_argument(
        "--out",
        default="yolo/dataset",
        help="Output dataset root",
    )
    p_prep.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training split ratio",
    )
    p_prep.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio",
    )
    p_prep.add_argument("--move", action="store_true", help="Move files instead of copy")
    p_prep.set_defaults(
        func=lambda a: prepare_dataset(
            Path(a.images), Path(a.labels), Path(a.out), a.train_ratio, a.val_ratio, a.move
        )
    )

    p_train = sub.add_parser("train", help="Train a YOLO model")
    p_train.add_argument("--data", default="dataset.yaml", help="Dataset YAML")
    p_train.add_argument("--model", default="yolov8n.pt", help="Base model path")
    p_train.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    p_train.add_argument("--imgsz", type=int, default=640, help="Image size")
    p_train.add_argument("--batch", type=int, default=16, help="Batch size")
    p_train.add_argument("--device", default=DEFAULT_DEVICE, help="Device (e.g. '0' for first GPU, 'cpu')")
    p_train.set_defaults(func=lambda a: train_model(Path(a.data), a.model, a.epochs, a.imgsz, a.batch, a.device))

    p_pred = sub.add_parser("predict", help="Run inference")
    p_pred.add_argument("source", help="Image or directory")
    p_pred.add_argument("--weights", help="Path to weights (defaults to latest)")
    p_pred.add_argument("--imgsz", type=int, default=640, help="Image size")
    p_pred.add_argument("--device", default=DEFAULT_DEVICE, help="Device (e.g. '0' for first GPU, 'cpu')")
    p_pred.set_defaults(func=lambda a: predict_image(Path(a.source), Path(a.weights) if a.weights else None, a.imgsz, a.device))

    p_test = sub.add_parser("test", help="Evaluate on the test split")
    p_test.add_argument("--data", default="dataset.yaml", help="Dataset YAML")
    p_test.add_argument("--weights", help="Path to weights (defaults to latest)")
    p_test.add_argument("--imgsz", type=int, default=640, help="Image size")
    p_test.add_argument("--device", default=DEFAULT_DEVICE, help="Device (e.g. '0' for first GPU, 'cpu')")
    p_test.set_defaults(
        func=lambda a: test_model(Path(a.data), Path(a.weights) if a.weights else None, a.imgsz, a.device)
    )

    p_cards = sub.add_parser("cards", help="Export detected objects as cropped images")
    p_cards.add_argument("source", help="Image or directory")
    p_cards.add_argument("--out", default="cards", help="Destination directory")
    p_cards.add_argument("--weights", help="Path to weights (defaults to latest)")
    p_cards.add_argument("--imgsz", type=int, default=640, help="Image size")
    p_cards.add_argument("--device", default=DEFAULT_DEVICE, help="Device (e.g. '0' for first GPU, 'cpu')")
    p_cards.set_defaults(
        func=lambda a: export_cards(
            Path(a.source), Path(a.out), Path(a.weights) if a.weights else None, a.imgsz, a.device
        )
    )

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
