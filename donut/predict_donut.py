#!/usr/bin/env python
"""Run inference с детальным логом предикта."""
import sys
import json
from PIL import Image
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

def main(model_dir: str, img_path: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = DonutProcessor.from_pretrained(model_dir, use_fast=True)
    model     = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device)

    # 1) Подготовка изображения
    image = Image.open(img_path).convert("RGB")
    pixel_values = processor.image_processor(
        images=image,
        return_tensors="pt"
    ).pixel_values.to(device)

    # 2) Стартовые токены
    bos_id = processor.tokenizer.bos_token_id
    decoder_input_ids = torch.tensor([[bos_id]], device=device)

    # 3) Генерация
    generated = model.generate(
        pixel_values=pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=processor.tokenizer.model_max_length,
        num_beams=5,
        early_stopping=True,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
        use_cache=True,
    )

    # 4) Debug: распечатаем сами ID и raw-декод
    print(">>> Generated token IDs:", generated.tolist())
    raw_pred = processor.batch_decode(generated, skip_special_tokens=False)[0]
    print(">>> Raw prediction (with all specials):")
    print(repr(raw_pred))

    # 5) Отчистка от специальных токенов
    pred = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    print(">>> Clean prediction (skip_special_tokens=True):")
    print(repr(pred))

    # 6) Пытаемся спарсить
    try:
        result = json.loads(pred)
    except json.JSONDecodeError:
        result = {"error": "failed to parse", "raw": pred}

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict_donut.py <model_dir> <image_path>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
