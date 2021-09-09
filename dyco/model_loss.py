from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from model_utils import EPSILON


class BCEWOLabelsLoss(nn.Module):

    def __init__(self, with_logits=False):
        super().__init__()
        self.with_logits = with_logits
        self.eps = EPSILON()

    def forward(self, pos_preds: Tensor, neg_preds: Tensor):
        if self.with_logits:
            pos_preds, neg_preds = torch.sigmoid(pos_preds), torch.sigmoid(neg_preds)
        pos_loss = -torch.log(pos_preds + self.eps).mean()
        neg_loss = -torch.log(1 - neg_preds + self.eps).mean()
        return pos_loss + neg_loss

    def __repr__(self):
        return f"{self.__class__.__name__}(with_logits={self.with_logits})"
