#!/usr/bin/env python
"""Run inference with a fine-tuned Donut model."""
import sys
import json
from PIL import Image
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

TASK_PROMPT = "<s_serialize>"

def main(model_dir: str, img_path: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Загрузка процессора и модели
    processor = DonutProcessor.from_pretrained(model_dir, use_fast=True)
    model     = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device)

    # 2) Читаем и конвертим изображение
    image = Image.open(img_path).convert("RGB")

    # 3) Готовим входы точно так же, как при обучении:
    #    передаём и картинку, и префикс TASK_PROMPT
    inputs = processor(
        images=image,
        text=TASK_PROMPT,
        return_tensors="pt"
    ).to(device)

    # 4) Генерируем
    generated_ids = model.generate(
        **inputs,
        max_length=processor.tokenizer.model_max_length,
        num_beams=5,
        early_stopping=True,
        use_cache=True,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    # 5) Декодируем — сначала с keep_special, чтобы увидеть, что модель реально генерит
    raw = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    print(">>> raw with specials:", repr(raw))

    # 6) А потом без спецтокенов
    cleaned = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(">>> without specials:", repr(cleaned))

    # 7) Парсим JSON
    try:
        output = json.loads(cleaned)
    except json.JSONDecodeError:
        output = {"error": "failed to parse", "raw": cleaned}

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict_donut.py <model_dir> <image_path>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
