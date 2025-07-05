# Computer Vision Pipelines

This repository contains two simple command line tools to work with product card images.

- **YOLOv8 pipeline** (`yolo/pipeline.py`) – dataset preparation and training for object detection.
- **Donut pipeline** (`donut/pipeline.py`) – preparing OCR data from cropped card images.

## Installation

```bash
pip install -r requirements.txt
```

## YOLO Usage

1. Put your images in `yolo/images/` and matching YOLO labels in `yolo/labels/`.
2. Create the dataset structure:
   ```bash
   python pipeline.py prepare
   ```
   The dataset will be stored in `yolo/dataset/`.
3. Train the model:
   ```bash
   python pipeline.py train --model yolov8n.pt --epochs 50
   ```
4. Predict on an image or folder:
   ```bash
   python pipeline.py predict path/to/image.jpg
   ```
   If `--weights` is not specified, the latest weights from `yolo/runs/` are used.
5. Export detected products as cropped images:
   ```bash
   python pipeline.py cards path/to/images --out cards/
   ```
6. Evaluate on the test split:
   ```bash
   python pipeline.py test
   ```

Edit `dataset.yaml` to list your class names.

## Donut Usage

Card images cropped by YOLO can be prepared for Donut OCR training.
Ground truth files must be JSON next to each image. Examples of `promo_type`
and fields:

```jsonc
{
  "promo_type": "percent_discount",
  "new_price": 14.99,
  "old_price": 29.99,
  "discount_pct": 50,
  "unit": "kg"
}
```

Run the preparer to split cards into train/val/test (defaults to `cards/`):

```bash
python donut/pipeline.py prepare
```

The resulting dataset will be created in `donut_dataset/`.
