# YOLOv8 Pipeline

Simple tools to prepare a dataset, train a YOLOv8 model and run predictions.

## Usage

1. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
2. Put your images in `images/` and matching YOLO labels in `labels/`.
3. Create the dataset structure with train/val/test splits
   ```bash
   python pipeline.py prepare
   ```
4. Train the model
   ```bash
   python pipeline.py train --model yolov8n.pt --epochs 50
   ```
5. Predict on an image or folder
   ```bash
   python pipeline.py predict path/to/image.jpg
   ```
   If `--weights` is not specified, the latest weights from `runs/train/` are used.

6. Evaluate on the test split
   ```bash
   python pipeline.py test
   ```

Edit `dataset.yaml` to list your class names.
