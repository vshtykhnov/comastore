import argparse
import json
import os
from PIL import Image
from transformers import VisionEncoderDecoderModel, DonutProcessor
from datasets import Dataset
from pdf2image import convert_from_path
import torch


def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def collate_fn(processor, batch):
    images = [Image.open(b['file_name']).convert('RGB') for b in batch]
    pixel_values = processor(images=images, return_tensors='pt').pixel_values
    labels = processor.tokenizer(
        [json.dumps(b['ground_truth'], ensure_ascii=False) for b in batch],
        return_tensors='pt',
        padding='max_length',
        truncation=True
    ).input_ids
    labels[labels == processor.tokenizer.pad_token_id] = -100
    return {'pixel_values': pixel_values, 'labels': labels}


def train(args):
    processor = DonutProcessor.from_pretrained('naver-clova-ix/donut-base')
    model = VisionEncoderDecoderModel.from_pretrained('naver-clova-ix/donut-base')

    train_data = Dataset.from_list(load_jsonl(args.train))
    val_data = Dataset.from_list(load_jsonl(args.val))

    def collate(batch):
        return collate_fn(processor, batch)

    from transformers import TrainingArguments, Trainer

    training_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=args.epochs,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_steps=500,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=collate
    )

    trainer.train()
    trainer.save_model(args.out)
    processor.save_pretrained(args.out)


def parse(args):
    processor = DonutProcessor.from_pretrained(args.model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_dir)
    if torch.cuda.is_available():
        model.to('cuda')

    pages = convert_from_path(args.pdf, dpi=300)
    offers = []
    for idx, image in enumerate(pages, start=1):
        inputs = processor(image.convert('RGB'), return_tensors='pt').pixel_values
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
        with torch.no_grad():
            generated = model.generate(inputs, max_length=512)
        json_str = processor.batch_decode(generated, skip_special_tokens=True)[0]
        data = json.loads(json_str)
        valid_from = data.get('valid_from') or data.get('ground_truth', {}).get('valid_from')
        valid_to = data.get('valid_to') or data.get('ground_truth', {}).get('valid_to')
        offers_list = data.get('offers') or data.get('ground_truth', {}).get('offers', [])
        for off in offers_list:
            off['page'] = idx
            if valid_from:
                off['valid_from'] = valid_from
            if valid_to:
                off['valid_to'] = valid_to
            offers.append(off)

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(offers, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Biedronka leaflet parser')
    subparsers = parser.add_subparsers(dest='command')

    t = subparsers.add_parser('train', help='Fine-tune model')
    t.add_argument('--train', required=True, help='Path to train.jsonl')
    t.add_argument('--val', required=True, help='Path to val.jsonl')
    t.add_argument('--out', required=True, help='Output directory for model')
    t.add_argument('--epochs', type=int, default=5)

    p = subparsers.add_parser('parse', help='Parse PDF with trained model')
    p.add_argument('pdf', help='PDF file')
    p.add_argument('model_dir', help='Path to trained model directory')
    p.add_argument('--out', default='offers.json', help='Output JSON file')

    args = parser.parse_args()

    if args.command == 'train':
        train(args)
    elif args.command == 'parse':
        parse(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
