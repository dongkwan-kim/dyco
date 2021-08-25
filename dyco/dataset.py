import os
from pprint import pprint
from typing import Union, List

import torch

from torch_geometric.datasets import JODIEDataset
from torch_geometric.datasets import BitcoinOTC
from torch_geometric.datasets import ICEWS18, GDELT
from torch_geometric.data import TemporalData
from torch_geometric.datasets.icews import EventDataset

from ogb.nodeproppred import PygNodePropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset
from ogb.nodeproppred import Evaluator as NodeEvaluator
from ogb.linkproppred import Evaluator as LinkEvaluator

from data_utils import from_events_to_singleton_data, ToTemporalData


class SingletonICEWS18(ICEWS18):

    def __init__(self, root, split='train', transform=None, pre_transform=None, pre_filter=None):
        self.split = split
        super(SingletonICEWS18, self).__init__(root, split, transform, pre_transform, pre_filter)

    def download(self):
        return super(SingletonICEWS18, self).download()

    def process(self):
        s = self.splits
        data_list = super(ICEWS18, self).process()
        torch.save(self.collate([from_events_to_singleton_data(data_list[s[0]:s[1]])]), self.processed_paths[0])
        torch.save(self.collate([from_events_to_singleton_data(data_list[s[1]:s[2]])]), self.processed_paths[1])
        torch.save(self.collate([from_events_to_singleton_data(data_list[s[2]:s[3]])]), self.processed_paths[2])

    def __repr__(self):
        return "{}(split={})".format(self.__class__.__name__, self.split)


class SingletonGDELT(GDELT):

    def __init__(self, root, split='train', transform=None, pre_transform=None, pre_filter=None):
        self.url = 'https://github.com/INK-USC/RE-Net/raw/master/data/GDELT'
        self.split = split
        super(SingletonGDELT, self).__init__(root, split, transform, pre_transform, pre_filter)

    def download(self):
        return super(SingletonGDELT, self).download()

    def process(self):
        s = self.splits
        data_list = super(GDELT, self).process()
        torch.save(self.collate([from_events_to_singleton_data(data_list[s[0]:s[1]])]), self.processed_paths[0])
        torch.save(self.collate([from_events_to_singleton_data(data_list[s[1]:s[2]])]), self.processed_paths[1])
        torch.save(self.collate([from_events_to_singleton_data(data_list[s[2]:s[3]])]), self.processed_paths[2])

    def __repr__(self):
        return "{}(split={})".format(self.__class__.__name__, self.split)


def _get_dataset_at_cls_dir(cls, path, transform=None, pre_transform=None, *args, **kwargs):
    path_with_name = os.path.join(path, cls.__name__)
    if cls in [PygNodePropPredDataset, PygLinkPropPredDataset]:
        return cls(root=path_with_name, transform=transform, pre_transform=pre_transform, *args, **kwargs)
    else:
        return cls(path_with_name, transform=transform, pre_transform=pre_transform, *args, **kwargs)


def get_dynamic_graph_dataset(path, name: str, transform=None, pre_transform=None, *args, **kwargs):
    if name.startswith("JODIEDataset"):
        _, sub_name = name.split("/")
        jodie_dataset = _get_dataset_at_cls_dir(JODIEDataset, path, transform=transform, pre_transform=pre_transform,
                                                name=sub_name, *args, **kwargs)
        jodie_dataset.num_nodes = jodie_dataset[0].dst.max().item() + 1
        return jodie_dataset
    elif name in ["SingletonICEWS18", "SingletonGDELT"]:
        args_wo_split = (eval(name), path, *args)
        kwargs_wo_split = dict(transform=transform, pre_transform=pre_transform, **kwargs)
        train_set = _get_dataset_at_cls_dir(*args_wo_split, **kwargs_wo_split, split="train")
        val_set = _get_dataset_at_cls_dir(*args_wo_split, **kwargs_wo_split, split="val")
        test_set = _get_dataset_at_cls_dir(*args_wo_split, **kwargs_wo_split, split="test")
        return train_set, val_set, test_set
    elif name in ["BitcoinOTC"]:
        # todo: train-val-test split
        # return _get_dataset_at_cls_dir(eval(name), path, transform=transform, pre_transform=pre_transform,
        #                                *args, **kwargs)
        raise NotImplementedError
    elif name.startswith("ogb"):
        cls = PygNodePropPredDataset if name.startswith("ogbn") else PygLinkPropPredDataset
        return _get_dataset_at_cls_dir(cls, path, transform=transform, pre_transform=pre_transform,
                                       name=name, *args, **kwargs)
    else:
        raise ValueError("Wrong name: {}".format(name))


if __name__ == '__main__':
    PATH = "/mnt/nas2/GNN-DATA/PYG/"
    NAME = "SingletonICEWS18"
    # JODIEDataset/reddit, JODIEDataset/wikipedia, JODIEDataset/mooc, JODIEDataset/lastfm
    #   TemporalData(dst=[157474], msg=[157474, 172], src=[157474], t=[157474], y=[157474])
    # ogbn-arxiv, ogbl-collab, ogbl-citation2
    #   Data(edge_index=[2, 1166243], node_year=[169343, 1], x=[169343, 128], y=[169343, 1])
    #   Data(edge_index=[2, 2358104], edge_weight=[2358104, 1], edge_year=[2358104, 1], x=[235868, 128])
    #   Data(edge_index=[2, 30387995], node_year=[2927963, 1], x=[2927963, 128])
    # SingletonICEWS18, SingletonGDELT
    #   Data(edge_index=[2, 373018], rel=[373018, 1], t=[373018, 1])
    #   Data(edge_index=[2, 1734399], rel=[1734399, 1], t=[1734399, 1])
    # BitcoinOTC
    #   Data(edge_attr=[148], edge_index=[2, 148])
    _dataset = get_dynamic_graph_dataset(
        PATH, NAME,
        # transform=ToTemporalData(),  # if necessary.
    )
    if isinstance(_dataset, tuple):
        _dataset = _dataset[0]
    print(_dataset)
    for d in _dataset:
        print(d)

    if NAME.startswith("Singleton"):
        _data = _dataset[0]
        print(torch.unique(_data.t))

    if NAME.startswith("JODIEDataset"):
        from collections import Counter
        _data = _dataset[0]
        print(torch.unique(_data.t))
        for i, (k, v) in enumerate(Counter(_data.t.tolist()).most_common()):
            print(k, v)
            if i > 20:
                break

    if NAME == "ogbl-collab":
        _split_edge = _dataset.get_edge_split()
        _pos_train_edge = _split_edge['train']['edge']
        _pos_valid_edge = _split_edge['valid']['edge']
        _neg_valid_edge = _split_edge['valid']['edge_neg']
        _pos_test_edge = _split_edge['test']['edge']
        _neg_test_edge = _split_edge['test']['edge_neg']
        assert _pos_train_edge.size() == torch.Size([1179052, 2])
        assert torch.allclose(_pos_train_edge.float().std(), _dataset[0].edge_index.float().std())
