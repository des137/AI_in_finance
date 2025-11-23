from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForSequenceClassification

from .base_model import BaseModel

@dataclass
class TextClassificationConfig:
    model_name: str
    num_labels: int
    freeze_base: bool = False

class TextClassificationModel(BaseModel):
    def __init__(self, cfg: TextClassificationConfig):
        super().__init__()
        hf_config = AutoConfig.from_pretrained(
            cfg.model_name,
            num_labels=cfg.num_labels,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name,
            config=hf_config,
        )

        if cfg.freeze_base:
            for param in self.model.base_model.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }
