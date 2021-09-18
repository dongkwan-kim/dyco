from abc import ABC, abstractmethod
from pprint import pprint
from typing import List, Dict, Callable, Iterable, Any, Union
from functools import reduce

from ogb.nodeproppred import Evaluator as OGBNodeEvaluator
from ogb.linkproppred import Evaluator as OGBLinkEvaluator
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor

from utils import merge_dict, try_get_from_dict

EvaluatorType = Union[OGBNodeEvaluator, OGBLinkEvaluator, Any]
VGETransformToIterate = Callable[[EvaluatorType], Iterable[EvaluatorType]]
VGETransformToReduce = Callable[[List[Dict]], Dict]


class VersatileLinkEvaluator(ABC):

    def __init__(self, name, metrics=None, k_list=None):
        self.metrics = metrics or []
        self.k_list = k_list

    @abstractmethod
    def eval(self, input_dict: Dict[str, Tensor]) -> Dict[str, float]:
        pass

    def _eval_mrr_and_hits(self, logits, y):
        """Given ground-truth :obj:`y`, computes Mean Reciprocal Rank (MRR) and Hits at 1/3/10.
        From https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/models/re_net.py#L195
        """
        _, perm = logits.sort(dim=1, descending=True)
        mask = (y.view(-1, 1) == perm)

        metric_values = []

        if "mrr" in self.metrics:
            nnz = mask.nonzero(as_tuple=False)
            mrr = (1 / (nnz[:, -1] + 1).to(torch.float)).mean().item()
            metric_values.append(mrr)

        if "hits" in self.metrics:
            assert self.k_list is not None
            # e.g., hits3 = mask[:, :3].sum().item() / y.size(0)
            hit_list_at_k = [(mask[:, :k].sum().item() / y.size(0)) for k in self.k_list]
            metric_values += hit_list_at_k

        return torch.tensor(metric_values)

    def _eval_ap_and_auc(self, logits, y):
        assert y.max().item() <= 1
        metric_values = []

        if "ap" in self.metrics:
            metric_values.append(average_precision_score(y, logits))

        if "auc" in self.metrics:
            metric_values.append(roc_auc_score(y, logits))

        return torch.tensor(metric_values)


class TKGEvaluator(VersatileLinkEvaluator):

    def __init__(self, name, k_list=None):
        metrics = ["mrr", "hits"]
        k_list = k_list or [1, 3, 10]
        super().__init__(name, metrics=metrics, k_list=k_list)

    def _parse_and_check_input(self, input_dict: Dict[str, Tensor]):
        assert "obj_pred" in input_dict
        assert "obj_node" in input_dict
        assert "sub_pred" in input_dict
        assert "sub_node" in input_dict
        return try_get_from_dict(input_dict, ["obj_pred", "obj_node", "sub_pred", "sub_node"],
                                 iter_all=True, as_dict=False)

    def eval(self, input_dict: Dict[str, Tensor]) -> Dict[str, float]:
        obj_pred, obj_node, sub_pred, sub_node = self._parse_and_check_input(input_dict)
        obj_node_size, sub_node_size = obj_node.size(0), sub_node.size(0)

        obj_metric_values = self._eval_mrr_and_hits(obj_pred, obj_node) * obj_node_size
        sub_metric_values = self._eval_mrr_and_hits(sub_pred, sub_node) * sub_node_size

        names = ["mrr"] + [f"hits@{k}" for k in self.k_list]
        values = (obj_metric_values + sub_metric_values) / (obj_node_size + sub_node_size)
        return dict(zip(names, values.tolist()))


class VersatileGraphEvaluator:

    def __init__(self, name: str,
                 transform_to_iterate: VGETransformToIterate = None,
                 transform_to_reduce: VGETransformToReduce = None,
                 *args, **kwargs):
        self.name = name
        self.transform_to_generate = transform_to_iterate or (lambda vge: [vge])
        self.transform_to_reduce = transform_to_reduce or merge_dict
        if name.startswith("ogbn"):
            self.evaluator = OGBNodeEvaluator(name)
        elif name.startswith("ogbl"):
            self.evaluator = OGBLinkEvaluator(name)
        elif name.startswith("Singleton"):
            self.evaluator = TKGEvaluator(name, *args, **kwargs)
        elif name.startswith("JODIEDataset"):
            raise NotImplementedError
        elif name == "BitcoinOTC":
            raise NotImplementedError
        else:
            raise ValueError("Wrong name: {}".format(name))

    def eval(self, input_dict: Dict[str, Tensor]):
        eval_outs = []
        for evaluator in self.transform_to_generate(self.evaluator):
            eval_outs.append(evaluator.eval(input_dict))
        return self.transform_to_reduce(eval_outs)

    @staticmethod
    def set_hits_k(k_list: List[int]) -> VGETransformToIterate:
        """
        :param k_list: List of ks
        :return:
        """

        def _set_hits_k(evaluator):
            for k in k_list:
                evaluator.K = k
                yield evaluator

        return _set_hits_k


if __name__ == '__main__':
    import numpy as np
    import torch

    torch.manual_seed(0)
    np.random.seed(0)

    _evaluator = VersatileGraphEvaluator(
        name='ogbl-collab',
        transform_to_iterate=VersatileGraphEvaluator.set_hits_k([10, 50, 100]),
    )
    y_pred_pos = torch.tensor(np.random.randn(1000, ))
    y_pred_neg = torch.tensor(np.random.randn(1000))
    result = _evaluator.eval({'y_pred_pos': y_pred_pos, 'y_pred_neg': y_pred_neg})
    pprint(result)

    _evaluator = VersatileGraphEvaluator(
        name='SingletonGDELT',
    )
    N, C = 100, 7
    _obj_pred = torch.randn((N, C))
    _obj_node = torch.randint(C, (N,))
    _sub_pred = torch.randn((N, C))
    _sub_node = torch.randint(C, (N,))
    result = _evaluator.eval({
        "obj_pred": _obj_pred, "obj_node": _obj_node,
        "sub_pred": _sub_pred, "sub_node": _sub_node,
    })
    pprint(result)
