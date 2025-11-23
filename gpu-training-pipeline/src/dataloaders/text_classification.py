from dataclasses import dataclass
from typing import Tuple

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

@dataclass
class DataConfig:
    name: str
    text_field: str
    label_field: str
    max_length: int
    batch_size: int
    num_workers: int
    subset_ratio: float = 1.0

def get_dataloaders(cfg: DataConfig, model_name: str) -> Tuple[DataLoader, DataLoader]:
    dataset = load_dataset(cfg.name)

    if cfg.subset_ratio < 1.0:
        for split in dataset.keys():
            dataset[split] = dataset[split].shuffle(seed=42).select(
                range(int(len(dataset[split]) * cfg.subset_ratio))
            )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def preprocess(batch):
        encodings = tokenizer(
            batch[cfg.text_field],
            truncation=True,
            padding="max_length",
            max_length=cfg.max_length,
        )
        encodings["labels"] = batch[cfg.label_field]
        return encodings

    encoded = dataset.map(preprocess, batched=True)

    encoded.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    train_loader = DataLoader(
        encoded["train"],
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    test_split = "test" if "test" in encoded.keys() else "validation"
    eval_loader = DataLoader(
        encoded[test_split],
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    return train_loader, eval_loader
