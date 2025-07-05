#!/usr/bin/env python
"""Fine-tune Donut on the prepared dataset."""
import json
from pathlib import Path
from typing import Dict
from PIL import Image

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

    def __init__(self, images_dir: str | Path, gt_dir: str | Path, processor: DonutProcessor, max_length: int = 512) -> None:
        self.images = sorted(Path(images_dir).glob("*"))
        self.gts: Dict[str, Path] = {}
        for p in Path(gt_dir).glob("*.json*"):
            self.gts[p.stem] = p
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.images[idx]
        key = img_path.stem
        with open(self.gts[key], encoding="utf-8") as f:
            gt = json.load(f)
        target_str = json.dumps(gt, ensure_ascii=False)
        image = Image.open(img_path).convert("RGB")
        data = self.processor(
            images=image,
            text=target_str,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        return {k: v.squeeze(0) for k, v in data.items()}


def donut_collate_fn(batch):
    """Merge a list of samples into a batch expected by VisionEncoderDecoderModel."""
    import torch

    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {
        "pixel_values": pixel_values,
        "labels": labels,
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
    processor.tokenizer.model_max_length = 128

    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    model.gradient_checkpointing_enable()

    train_ds = DonutDataset(train_images, train_gts, processor)
    val_ds = DonutDataset(val_images, val_gts, processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=1,
        predict_with_generate=True,
        eval_strategy="epoch",
        logging_steps=100,
        save_strategy="epoch",
        num_train_epochs=max_epochs,
        learning_rate=lr,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
    )

    torch.cuda.empty_cache()

    data_collator = donut_collate_fn

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor.tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)


if __name__ == "__main__":
    main()