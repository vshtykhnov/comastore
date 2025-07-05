#!/usr/bin/env python
"""Run inference with a fine-tuned Donut model."""
import sys
import json
from pathlib import Path
from PIL import Image
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel


def main(model_dir: str, img_path: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = DonutProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device)

    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    generated_ids = model.generate(
        **inputs,
        max_length=512,
        num_beams=5,
        early_stopping=True,
        use_cache=False,
    )
    pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    try:
        result = json.loads(pred)
    except json.JSONDecodeError:
        result = {"error": "failed to parse", "raw": pred}

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict_donut.py <model_dir> <image_path>")
        raise SystemExit(1)
    main(sys.argv[1], sys.argv[2])