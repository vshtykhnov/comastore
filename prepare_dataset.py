#!/usr/bin/env python
"""prepare_dataset.py

Create a YOLO-friendly dataset directory structure from existing `images/` and
`labels/` folders.

It matches each label file in `labels/` (e.g. `cat_01.txt`) with an image that
has the same stem (e.g. `cat_01.jpg` or `cat_01.png`) inside `images/`.
Then it randomly splits the pairs into train/val subsets (default 80/20) and
copies the files into:

    dataset/
      images/
        train/  *.jpg|*.png
        val/    *.jpg|*.png
      labels/
        train/  *.txt
        val/    *.txt

Usage:
  python prepare_dataset.py               # default 80/20 split
  python prepare_dataset.py --ratio 0.9   # 90% train, 10% val
  python prepare_dataset.py --move        # move instead of copy (saves disk)

After running, add a `dataset.yaml` like:

train: ./dataset/images/train
val:   ./dataset/images/val
names: [class0, class1, ...]
"""
import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_corresponding_image(label_path: Path, images_dir: Path) -> Path | None:
    """Return image file matching the label stem or None if not found."""
    stem = label_path.stem
    for ext in IMAGE_EXTS:
        img_path = images_dir / f"{stem}{ext}"
        if img_path.exists():
            return img_path
    return None


def prepare_dataset(images_dir: Path, labels_dir: Path, out_root: Path, ratio: float, move: bool):
    random.seed(42)
    out_images_train = out_root / "images" / "train"
    out_images_val = out_root / "images" / "val"
    out_labels_train = out_root / "labels" / "train"
    out_labels_val = out_root / "labels" / "val"
    for d in [out_images_train, out_images_val, out_labels_train, out_labels_val]:
        d.mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[Path, Path]] = []
    for label_path in labels_dir.glob("*.txt"):
        img_path = find_corresponding_image(label_path, images_dir)
        if img_path is None:
            print(f"[WARN] image for label '{label_path.name}' not found – skipping")
            continue
        pairs.append((img_path, label_path))

    if not pairs:
        print("No image/label pairs found – nothing to do")
        return

    random.shuffle(pairs)
    split_idx = int(len(pairs) * ratio)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    def transfer(src: Path, dst: Path):
        if move:
            shutil.move(src, dst)
        else:
            shutil.copy2(src, dst)

    for img_path, label_path in train_pairs:
        transfer(img_path, out_images_train / img_path.name)
        transfer(label_path, out_labels_train / label_path.name)

    for img_path, label_path in val_pairs:
        transfer(img_path, out_images_val / img_path.name)
        transfer(label_path, out_labels_val / label_path.name)

    print(f"Done! Train: {len(train_pairs)}, Val: {len(val_pairs)}")


def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset structure")
    parser.add_argument("--images", default="images", help="Source images directory")
    parser.add_argument("--labels", default="labels", help="Source labels directory")
    parser.add_argument("--out", default="dataset", help="Output dataset root directory")
    parser.add_argument("--ratio", type=float, default=0.8, help="Train split ratio (0-1)")
    parser.add_argument("--move", action="store_true", help="Move files instead of copy")
    args = parser.parse_args()

    prepare_dataset(Path(args.images), Path(args.labels), Path(args.out), args.ratio, args.move)


if __name__ == "__main__":
    main() 