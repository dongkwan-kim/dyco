import os
from pprint import pprint
from typing import Union, List

import torch

from torch_geometric.datasets import JODIEDataset
from torch_geometric.datasets import BitcoinOTC
from torch_geometric.datasets import ICEWS18, GDELT

from ogb.nodeproppred import PygNodePropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset
from ogb.nodeproppred import Evaluator as NodeEvaluator
from ogb.linkproppred import Evaluator as LinkEvaluator

from torch_geometric.data import TemporalData
from torch_geometric.datasets.icews import EventDataset


def _get_dataset_at_cls_dir(cls, path, *args, **kwargs):
    path_with_name = os.path.join(path, cls.__name__)
    if cls in [PygNodePropPredDataset, PygLinkPropPredDataset]:
        return cls(root=path_with_name, **kwargs)
    else:
        return cls(path_with_name, *args, **kwargs)


def get_dynamic_dataset(path, name: str, *args, **kwargs):
    if name.startswith("JODIEDataset"):
        _, sub_name = name.split("/")
        return _get_dataset_at_cls_dir(JODIEDataset, path, sub_name, *args, **kwargs)
    elif name in ["BitcoinOTC", "ICEWS18", "GDELT"]:
        name: Union[BitcoinOTC, ICEWS18, GDELT]
        return _get_dataset_at_cls_dir(eval(name), path, *args, **kwargs)
    elif name.startswith("ogbn") or name.startswith("ogbl"):
        cls = PygNodePropPredDataset if name.startswith("ogbn") else PygLinkPropPredDataset
        return _get_dataset_at_cls_dir(cls, path, name=name, *args, **kwargs)
    else:
        raise ValueError("Wrong name: {}".format(name))


if __name__ == '__main__':
    PATH = "/mnt/nas2/GNN-DATA/PYG/"
    # JODIEDataset/reddit, JODIEDataset/wikipedia, JODIEDataset/mooc, JODIEDataset/lastfm
    #   TemporalData(dst=[157474], msg=[157474, 172], src=[157474], t=[157474], y=[157474])
    # ogbn-arxiv, ogbl-collab, ogbl-citation2
    #   Data(edge_index=[2, 1166243], node_year=[169343, 1], x=[169343, 128], y=[169343, 1])
    #   Data(edge_index=[2, 2358104], edge_weight=[2358104, 1], edge_year=[2358104, 1], x=[235868, 128])
    #   Data(edge_index=[2, 30387995], node_year=[2927963, 1], x=[2927963, 128])
    # ICEWS18, GDELT
    #   Data(obj=[1], rel=[1], sub=[1], t=[1])
    # BitcoinOTC
    #   Data(edge_attr=[148], edge_index=[2, 148])
    _dataset = get_dynamic_dataset(
        PATH, "ogbl-collab",
    )
