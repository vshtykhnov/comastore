import argparse
from pathlib import Path
from ultralytics import YOLO


def train(args: argparse.Namespace) -> None:
    """Train a YOLO model using the provided dataset."""
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
    )


def predict(args: argparse.Namespace) -> None:
    """Run inference using a trained model."""
    model = YOLO(args.weights)
    results = model(
        args.source,
        project=args.project,
        name=args.name,
        save=True,
        imgsz=args.imgsz,
        device=args.device,
    )
    if results:
        print(f"Results saved to {results[0].save_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or run inference with YOLO")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_p = subparsers.add_parser("train", help="Train a YOLO model")
    train_p.add_argument("--model", default="yolov8n.pt", help="Path to model or config")
    train_p.add_argument("--data", default="dataset.yaml", help="Dataset YAML file")
    train_p.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    train_p.add_argument("--imgsz", type=int, default=640, help="Image size")
    train_p.add_argument("--batch", type=int, default=16, help="Batch size")
    train_p.add_argument("--project", default="runs/train", help="Training output directory")
    train_p.add_argument("--name", default="exp", help="Run name")
    train_p.add_argument("--device", default="cpu", help="Device to train on")
    train_p.set_defaults(func=train)

    pred_p = subparsers.add_parser("predict", help="Run inference with a trained model")
    pred_p.add_argument("weights", help="Path to trained weights")
    pred_p.add_argument("source", help="Image or directory of images")
    pred_p.add_argument("--imgsz", type=int, default=640, help="Image size")
    pred_p.add_argument("--project", default="runs/predict", help="Prediction output directory")
    pred_p.add_argument("--name", default="exp", help="Run name")
    pred_p.add_argument("--device", default="cpu", help="Device for inference")
    pred_p.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
