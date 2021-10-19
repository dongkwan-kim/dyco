from pprint import pprint
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
        return self.get_loss(g, x, batch)

    def get_loss(self, anchor_g, samples_n, batch) -> torch.FloatTensor:
        sim = self._similarity(anchor_g, samples_n)  # [B, F], [N, F] --> [B, N]
        sim = sim / self.temperature
        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))

        pos_idx, counts_per_batch = self.pos_index(batch, return_counts_per_batch=True)

        # same as (log_prob * pos_mask).sum(dim=1) / torch.sum(pos_mask, dim=1)
        pos_log_prob = scatter(torch.take(log_prob, pos_idx), batch, dim=0, reduce="sum")
        loss = pos_log_prob / counts_per_batch
        return -loss.mean()

    def __repr__(self):
        return f"{self.__class__.__name__}(t={self.temperature})"

    @staticmethod
    def pos_index(batch: Tensor, return_counts_per_batch=False):
        """
        :param batch: e.g., [0, 0, 1, 1, 1, 2, 2]
        :param return_counts_per_batch:
        :return: the 1d index of pos_mask.
            e.g., [0*7+0, 0*7+1, 1*7+2, 1*7+3, 1*7+4, 2*7+5, 2*7+6],
            that is, 1d index of
                 [[ True,  True, False, False, False, False, False],
                  [False, False,  True,  True,  True, False, False],
                  [False, False, False, False, False,  True,  True]])
        """
        b_index, b_counts = torch.unique_consecutive(batch, return_counts=True)

        b_cum_counts = torch.cumsum(b_counts, dim=0)
        b_cum_counts = torch.roll(b_cum_counts, 1)  # [2, 5, 7] -> [7, 2, 5]
        b_cum_counts[0] = 0

        num_nodes = batch.size(0)
        start_at_rows = b_index * num_nodes + b_cum_counts

        sparse_mask_row_list = []
        for _start, _count in zip(start_at_rows.tolist(), b_counts.tolist()):
            sparse_mask_row_list.append(torch.arange(_count) + _start)
        sparse_pos_mask = torch.cat(sparse_mask_row_list)

        if not return_counts_per_batch:
            return sparse_pos_mask.to(batch.device)
        else:
            return sparse_pos_mask.to(batch.device), b_counts

    @staticmethod
    def pos_mask(batch: Tensor):
        """Practically not usable because of memory limits."""
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
        return self.get_loss(g, x, batch)

    def __repr__(self):
        return f"{self.__class__.__name__}(t={self.temperature}, ro={self.readout})"


class BCEWOLabelsLoss(nn.Module):

    def __init__(self, with_logits=False):
        super().__init__()
        self.with_logits = with_logits
        self.eps = EPSILON()

    @staticmethod
    def sigmoid_safe(x):
        return torch.sigmoid(x) if x is not None else None

    def forward(self, pos_preds: Tensor, neg_preds: Tensor):
        loss = 0
        if self.with_logits:
            pos_preds, neg_preds = self.sigmoid_safe(pos_preds), self.sigmoid_safe(neg_preds)
        if pos_preds is not None:
            loss += -torch.log(pos_preds + self.eps).mean()
        if neg_preds is not None:
            loss += -torch.log(1 - neg_preds + self.eps).mean()
        return loss

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

    elif MODE == "pos_index":
        _b = torch.Tensor([0, 0, 1, 1, 1, 2, 2]).long()
        print(InfoNCELoss.pos_index(_b))
