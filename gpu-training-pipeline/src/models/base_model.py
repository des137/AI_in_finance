from abc import ABC, abstractmethod
from typing import Dict

import torch
from torch import nn

class BaseModel(ABC, nn.Module):
    @abstractmethod
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
