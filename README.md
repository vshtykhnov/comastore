# Leaflet PDF ➜ YOLOv8 Pipeline

This repository contains small utilities to turn promotional leaflets (PDF) into a ready-to-train YOLOv8 dataset and to train/run inference.

## Scripts

| Script               | Purpose                                                                                                |
| -------------------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| `app.py`             | Convert every PDF in `raw_pdf/` into PNG pages inside `pages/` (default 72 dpi).                       |
| `detect_dpi.py`      | Print the effective DPI of each PDF page or, with `--images`, the DPI metadata of existing images.     |
| `prepare_dataset.py` | Match `labels/*.txt` with `images/\*.(png                                                              | jpg)`having the same stem, split into **train/val** and copy/move them into`dataset/` structure expected by Ultralytics. |
| `train_yolo.py`      | Minimal CLI wrapper around **Ultralytics YOLO** to train (`train` sub-command) or predict (`predict`). |

## Quick-start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Poppler is required on Windows for some PDF tools (e.g. `pdf2image`). Install it if you plan to use those tools.

### 2. Convert PDFs to images

Place source PDFs into `raw_pdf/` and run:

```bash
python app.py
```

Images are created in `pages/` with filenames like `flyer_p01.png`.

### 3. Annotate images

Label the images (e.g. with **LabelImg**, **Roboflow**, etc.) in **YOLO format** and save resulting `.txt` files to a `labels/` folder alongside the raw images in `images/`.

```
images/
  flyer_p01.png
  flyer_p02.png
labels/
  flyer_p01.txt
  flyer_p02.txt
```

### 4. Prepare dataset directory

```bash
python prepare_dataset.py --ratio 0.8   # 80 % train, 20 % val
```

This produces the structure:

```
dataset/
  images/train/*.png
  images/val/*.png
  labels/train/*.txt
  labels/val/*.txt
```

Create (or adjust) `dataset.yaml` to point to these folders:

```yaml
train: ./dataset/images/train
val: ./dataset/images/val
names: [item] # list your class names here
```

### 5. Train YOLOv8

```bash
python train_yolo.py train --model yolov8n.pt --data dataset.yaml --epochs 50 --batch 16 --imgsz 640 --device 0
```

Weights and logs are saved to `runs/train/<name>`.

### 6. Run inference

```bash
python train_yolo.py predict best.pt test_images/ --imgsz 640 --device 0
```

Predicted images are saved to `runs/predict/<name>`.

## License

MIT
