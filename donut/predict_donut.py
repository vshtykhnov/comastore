#!/usr/bin/env python
"""Run inference with a fine-tuned Donut model."""
import sys
import json
from PIL import Image
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

def main(model_dir: str, img_path: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = DonutProcessor.from_pretrained(model_dir, use_fast=True)
    model     = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device)

    image = Image.open(img_path).convert("RGB")
    pixel_values = processor.feature_extractor(
        images=image, return_tensors="pt"
    ).pixel_values.to(device)

    cls_id = processor.tokenizer.cls_token_id
    decoder_input_ids = torch.tensor([[cls_id]], device=device)

    generated_ids = model.generate(
        pixel_values=pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=processor.tokenizer.model_max_length,
        num_beams=5,
        early_stopping=True,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
        use_cache=True,
    )

    raw_pred = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    cls_tok = processor.tokenizer.cls_token or ""
    eos_tok = processor.tokenizer.eos_token or ""
    if raw_pred.startswith(cls_tok):
        raw_pred = raw_pred[len(cls_tok):]
    raw_pred = raw_pred.replace(eos_tok, "").strip()

    try:
        result = json.loads(raw_pred)
    except json.JSONDecodeError:
        result = {"error": "failed to parse", "raw": raw_pred}

    # 8) Выводим красиво
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict_donut.py <model_dir> <image_path>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
