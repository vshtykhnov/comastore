# Biedronka Leaflet Parser

This project provides a simple CLI to train a Donut model and parse promotional PDFs from the Biedronka supermarket. The goal is to extract all promotional offers into a single JSON file.

## Project Structure

```
.
├── app.py            # CLI application
├── data/             # Training and validation JSONL files
├── pages/            # PNG pages used for training
├── donut_out/        # Directory to store trained model
```

- `data/train.jsonl` and `data/val.jsonl` contain training examples. Each line is a JSON object describing one page.
- `pages/` contains the PNG images referenced in the JSONL files.

## Installation

Use Python 3.11 and install dependencies:

```bash
pip install "transformers>=4.44" datasets accelerate bitsandbytes
pip install pdf2image pillow torch torchvision
```

You also need `poppler` for `pdf2image` to convert PDFs to images.

## Training

Fine-tune the base Donut model using annotated pages:

```bash
python app.py train \
    --train data/train.jsonl \
    --val data/val.jsonl \
    --out donut_out \
    --epochs 5
```

After training, the `donut_out` directory will contain the model and processor weights.

## Parsing a PDF

To parse a PDF and extract all offers:

```bash
python app.py parse leaflet.pdf donut_out --out offers.json
```

This command will convert each PDF page to an image, run the model and save a flat list of offers to `offers.json`.

## Output Format

```
[
  {
    "page": 1,
    "name": "Czereśnie na wagę",
    "promo_type": "percent_discount",
    "new": 16.99,
    "old": 24.99,
    "discount_pct": 32,
    "unit": "kg",
    "valid_from": "2025-07-03",
    "valid_to": "2025-07-05"
  },
  ...
]
```

Each offer includes the page number and common fields such as `promo_type` and prices. If `valid_from` and `valid_to` are present on the page, they are attached to each offer.
