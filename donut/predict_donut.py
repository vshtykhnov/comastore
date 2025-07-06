#!/usr/bin/env python
"""Run inference with a fine-tuned Donut model."""
import sys
import json
from PIL import Image
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

def main(model_dir: str, img_path: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Загружаем processor и модель
    processor = DonutProcessor.from_pretrained(model_dir, use_fast=True)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device)

    # 2) Подготавливаем пиксели
    image = Image.open(img_path).convert("RGB")
    pixel_values = processor.image_processor(
        images=image,
        return_tensors="pt"
    ).pixel_values.to(device)

    # 3) Стартовый ID для декодера — bos_token_id, а не cls_token_id
    bos_id = processor.tokenizer.bos_token_id
    decoder_input_ids = torch.tensor([[bos_id]], device=device)

    # 4) Генерируем
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

    # 5) Декодируем И сразу пропускаем специальные токены
    #    это уберёт все <s>, </s>, <pad> и пр.
    raw = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # 6) Пытаемся распарсить JSON
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"error": "failed to parse", "raw": raw}

    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict_donut.py <model_dir> <image_path>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
