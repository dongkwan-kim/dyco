from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch_scatter import scatter

from model_utils import EPSILON, Readout


class InfoNCELoss(nn.Module):
    r"""InfoNCELoss: Mostly adopted from codes below
        - https://github.com/GraphCL/PyGCL/blob/main/GCL/losses/infonce.py
        - https://github.com/GraphCL/PyGCL/blob/main/GCL/models/samplers.py#L64-L65

    InfoNCELoss_* = - log [ exp(sim(g, n_*)/t) ] / [ \sum_i exp(sim(g, n_i)/t) ]
                  = - exp(sim(g, n_*)/t) + log [ \sum_i exp(sim(g, n_i)/t) ]
    """
    def __init__(self, temperature):
        """
        :param temperature: The MoCo paper uses 0.07, while SimCLR uses 0.5.
        """
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, g, x, batch):
        return self.get_loss(g, x, self._pos_mask(batch))

    def get_loss(self, anchor_g, samples_n, pos_mask) -> torch.FloatTensor:
        sim = self._similarity(anchor_g, samples_n)  # [B, F], [N, F] --> [B, N]
        sim = sim / self.temperature
        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob * pos_mask
        loss = loss.sum(dim=1) / torch.sum(pos_mask, dim=1)
        return -loss.mean()

    def __repr__(self):
        return f"{self.__class__.__name__}(t={self.temperature})"

    @staticmethod
    def _pos_mask(batch: Tensor):
        """
        :param batch: e.g., [0, 0, 1, 1, 1, 2, 2]
        :return: Tensor the size of which is [B, N]
            e.g., tensor([[ True,  True, False, False, False, False, False],
                          [False, False,  True,  True,  True, False, False],
                          [False, False, False, False, False,  True,  True]])
        """
        bools_one_hot = torch.eye(batch.size(0), dtype=torch.bool, device=batch.device)  # [N]
        pos_mask = scatter(bools_one_hot, batch, dim=0, reduce="sum")  # [B, N]
        return pos_mask

    @staticmethod
    def _similarity(h1: torch.Tensor, h2: torch.Tensor):
        h1 = F.normalize(h1)  # [B, F]
        h2 = F.normalize(h2)  # [N, F]
        return h1 @ h2.t()  # [B, N]


class InfoNCEWithReadoutLoss(InfoNCELoss):

    def __init__(self, temperature, readout: Readout):
        super().__init__(temperature)
        self.readout = readout

    def forward(self, x, batch, *args):
        z_g, logits_g = self.readout(x, batch)
        g = z_g if logits_g is None else logits_g
        return self.get_loss(g, x, self._pos_mask(batch))

    def __repr__(self):
        return f"{self.__class__.__name__}(t={self.temperature}, ro={self.readout})"


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


if __name__ == '__main__':

    MODE = "InfoNCEWithReadoutLoss"

    if MODE == "InfoNCEWithReadoutLoss":
        _B = 3
        _N = 7
        _F = 5
        _g = torch.ones((_B, _F)).float()
        _n = torch.ones((_N, _F)).float()
        _b = torch.Tensor([0, 0, 1, 1, 1, 2, 2]).long()
        assert _b.size(-1) == _N
        assert _b.max().item() + 1 == _B

        _loss = InfoNCEWithReadoutLoss(
            0.5, Readout(readout_types="mean-max", use_in_mlp=False, use_out_linear=True,
                         hidden_channels=_F, out_channels=_F))
        print(_loss)
        print(_loss(_n, _b))
