#!/usr/bin/env python
"""Fine-tune Donut on the prepared dataset."""
import json
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import Dataset

from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)


class DonutDataset(Dataset):
    """Simple dataset reading images and JSON ground truth."""

    def __init__(self, images_dir: str | Path, gt_dir: str | Path, processor: DonutProcessor) -> None:
        self.images = sorted(Path(images_dir).glob("*"))
        self.gts: Dict[str, Path] = {p.stem: p for p in Path(gt_dir).glob("*.json")}
        self.processor = processor

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.images[idx]
        data = self.processor(images=img_path, return_tensors="pt")
        key = img_path.stem
        with open(self.gts[key], encoding="utf-8") as f:
            gt = json.load(f)
        target_str = json.dumps(gt, ensure_ascii=False)
        labels = self.processor.tokenizer(target_str, add_special_tokens=False, return_tensors="pt")
        input_ids = data["input_ids"].squeeze()
        pixel_values = data["pixel_values"].squeeze()
        labels_ids = labels["input_ids"].squeeze()
        return {
            "pixel_values": pixel_values,
            "labels": labels_ids,
            "attention_mask": input_ids.bool(),
        }


def main() -> None:
    train_images = "donut_dataset/images/train"
    train_gts = "donut_dataset/ground_truth/train"
    val_images = "donut_dataset/images/val"
    val_gts = "donut_dataset/ground_truth/val"
    output_dir = "donut_finetuned"
    model_name = "naver-clova-ix/donut-base"
    max_epochs = 5
    batch_size = 4
    lr = 5e-5

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = DonutProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)

    train_ds = DonutDataset(train_images, train_gts, processor)
    val_ds = DonutDataset(val_images, val_gts, processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
        evaluation_strategy="epoch",
        logging_steps=100,
        save_strategy="epoch",
        num_train_epochs=max_epochs,
        learning_rate=lr,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor.tokenizer,
        data_collator=lambda data: {
            "pixel_values": torch.stack([f["pixel_values"] for f in data]),
            "labels": torch.stack([f["labels"] for f in data]),
            "attention_mask": torch.stack([f["attention_mask"] for f in data]),
        },
    )

    trainer.train()
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
