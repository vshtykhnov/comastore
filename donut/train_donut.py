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
    """Reads images and JSON ground truth and returns pixel values + labels."""
    def __init__(
        self,
        images_dir: str | Path,
        gt_dir: str | Path,
        processor: DonutProcessor,
        max_length: int = 128,
    ) -> None:
        self.images = sorted(Path(images_dir).glob("*"))
        self.gts: Dict[str, Path] = {p.stem: p for p in Path(gt_dir).glob("*.json*")}
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.images[idx]
        gt = json.loads(Path(self.gts[img_path.stem]).read_text(encoding="utf-8"))

        # Dynamically use model's cls and eos tokens as prefix/suffix
        cls_token = self.processor.tokenizer.cls_token or ""
        eos_token = self.processor.tokenizer.eos_token or ""
        text = cls_token + json.dumps(gt, ensure_ascii=False) + eos_token

        image = Image.open(img_path).convert("RGB")
        encodings = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        # Return only the necessary fields
        return {
            "pixel_values": encodings.pixel_values.squeeze(0),
            "labels": encodings.input_ids.squeeze(0),
        }


def collate_fn(batch, pad_token_id: int):
    """Stack and mask padding tokens for labels."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    # mask padding
    labels = labels.masked_fill(labels == pad_token_id, -100)
    return {"pixel_values": pixel_values, "labels": labels}


def main() -> None:
    train_images = "donut_dataset/images/train"
    train_gts    = "donut_dataset/ground_truth/train"
    val_images   = "donut_dataset/images/val"
    val_gts      = "donut_dataset/ground_truth/val"
    output_dir   = "donut_finetuned"
    model_name   = "naver-clova-ix/donut-base"
    max_epochs   = 7
    lr           = 1e-5

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load processor and print special tokens for verification
    processor = DonutProcessor.from_pretrained(model_name)
    print("Using cls_token=", processor.tokenizer.cls_token)
    print("Using eos_token=", processor.tokenizer.eos_token)

    processor.tokenizer.model_max_length = 128

    # Load model and configure generation
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.gradient_checkpointing_enable()

    # Prepare datasets
    train_ds = DonutDataset(train_images, train_gts, processor)
    val_ds   = DonutDataset(val_images, val_gts, processor)

    pad_token_id = processor.tokenizer.pad_token_id

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=1,
        predict_with_generate=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        generation_max_length=processor.tokenizer.model_max_length,
        generation_num_beams=5,
        learning_rate=lr,
        num_train_epochs=max_epochs,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

    torch.cuda.empty_cache()

    # Trainer with custom collate
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda batch: collate_fn(batch, pad_token_id),
        tokenizer=processor.tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
