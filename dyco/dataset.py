import os
from pprint import pprint
from typing import Union, List, Dict, Type, Tuple

import torch

from torch_geometric.datasets import JODIEDataset
from torch_geometric.datasets import BitcoinOTC
from torch_geometric.datasets import ICEWS18, GDELT
from torch_geometric.data import TemporalData, Data, InMemoryDataset

from ogb.nodeproppred import PygNodePropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset

from data_utils import from_events_to_singleton_data, ToTemporalData


DatasetType = Union[Type[InMemoryDataset],
                    Tuple[Type[InMemoryDataset], Type[InMemoryDataset], Type[InMemoryDataset]]]


# noinspection PyUnresolvedReferences
class EventDatasetHelper:

    @property
    def processed_file_names(self):
        return ["train.pt", "val.pt", "test.pt", "train_val.pt", "train_val_test.pt"]

    def process(self):
        s = self.splits
        data_list = super(self.parent_dataset, self).process()
        torch.save(self.collate([from_events_to_singleton_data(data_list[s[0]:s[1]])]), self.processed_paths[0])
        torch.save(self.collate([from_events_to_singleton_data(data_list[s[1]:s[2]])]), self.processed_paths[1])
        torch.save(self.collate([from_events_to_singleton_data(data_list[s[2]:s[3]])]), self.processed_paths[2])
        torch.save(self.collate([from_events_to_singleton_data(data_list[s[0]:s[2]])]), self.processed_paths[3])
        torch.save(self.collate([from_events_to_singleton_data(data_list[s[0]:s[3]])]), self.processed_paths[4])

    def get_rel_split(self) -> Dict[str, slice]:
        s = self.splits
        return {"val": slice(s[1], s[2]), "test": slice(s[2], s[3])}

    def __repr__(self):
        return "{}(split={})".format(self.__class__.__name__, self.split)

    def add_mask(self, data):
        rel_split = self.get_rel_split()
        for k, v in sorted(rel_split.items()):
            if k in self.split:
                setattr(data, f"{k}_mask", rel_split[k])
                break
        return data


class SingletonICEWS18(EventDatasetHelper, ICEWS18):

    def __init__(self, root, split="train", transform=None, pre_transform=None, pre_filter=None):
        self.split = split
        self.parent_dataset = ICEWS18
        super(ICEWS18, self).__init__(root, transform, pre_transform, pre_filter)
        idx = self.processed_file_names.index("{}.pt".format(split))
        self.data, self.slices = torch.load(self.processed_paths[idx])

    def download(self):
        return super(SingletonICEWS18, self).download()

    def process(self):
        return super(SingletonICEWS18, self).process()

    def __getitem__(self, item):
        got = super(SingletonICEWS18, self).__getitem__(item)
        return self.add_mask(got)


class SingletonGDELT(EventDatasetHelper, GDELT):

    def __init__(self, root, split="train", transform=None, pre_transform=None, pre_filter=None):
        self.url = "https://github.com/INK-USC/RE-Net/raw/master/data/GDELT"
        self.split = split
        self.parent_dataset = GDELT
        super(GDELT, self).__init__(root, transform, pre_transform, pre_filter)
        idx = self.processed_file_names.index("{}.pt".format(split))
        self.data, self.slices = torch.load(self.processed_paths[idx])

    def download(self):
        return super(SingletonGDELT, self).download()

    def process(self):
        return super(SingletonGDELT, self).process()

    def __getitem__(self, item):
        got = super(SingletonGDELT, self).__getitem__(item)
        return self.add_mask(got)


def _get_dataset_at_cls_dir(cls, path, transform=None, pre_transform=None, *args, **kwargs):
    path_with_name = os.path.join(path, cls.__name__)
    if cls in [PygNodePropPredDataset, PygLinkPropPredDataset]:
        return cls(root=path_with_name, transform=transform, pre_transform=pre_transform, *args, **kwargs)
    else:
        return cls(path_with_name, transform=transform, pre_transform=pre_transform, *args, **kwargs)


def get_dynamic_graph_dataset(path, name: str, transform=None, pre_transform=None, *args, **kwargs) -> DatasetType:
    if name.startswith("JODIEDataset"):
        _, sub_name = name.split("/")
        jodie_dataset = _get_dataset_at_cls_dir(JODIEDataset, path, transform=transform, pre_transform=pre_transform,
                                                name=sub_name, *args, **kwargs)
        jodie_dataset.num_nodes = jodie_dataset.data.dst.max().item() + 1
        return jodie_dataset
    elif name in ["SingletonICEWS18", "SingletonGDELT"]:
        assert "splits" in kwargs
        splits: List[str] = kwargs.pop("splits")
        args_wo_split = (eval(name), path, *args)
        kwargs_wo_split = dict(transform=transform, pre_transform=pre_transform, **kwargs)
        train_set = _get_dataset_at_cls_dir(*args_wo_split, **kwargs_wo_split, split=splits[0])
        val_set = _get_dataset_at_cls_dir(*args_wo_split, **kwargs_wo_split, split=splits[1])
        test_set = _get_dataset_at_cls_dir(*args_wo_split, **kwargs_wo_split, split=splits[2])
        return train_set, val_set, test_set
    elif name in ["BitcoinOTC"]:
        # todo: train-val-test split
        # return _get_dataset_at_cls_dir(eval(name), path, transform=transform, pre_transform=pre_transform,
        #                                *args, **kwargs)
        raise NotImplementedError
    elif name.startswith("ogb"):
        cls = PygNodePropPredDataset if name.startswith("ogbn") else PygLinkPropPredDataset
        ogb_dataset = _get_dataset_at_cls_dir(cls, path, transform=transform, pre_transform=pre_transform,
                                              name=name, *args, **kwargs)
        ogb_dataset.num_nodes = ogb_dataset.data.x.size(0)
        return ogb_dataset
    else:
        raise ValueError("Wrong name: {}".format(name))


if __name__ == "__main__":
    PATH = "/mnt/nas2/GNN-DATA/PYG/"
    NAME = "ogbl-collab"
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
    _dataset_kwargs = {}
    if NAME.startswith("Singleton"):
        _dataset_kwargs["splits"] = ["train", "train_val", "train_val_test"]

    _dataset = get_dynamic_graph_dataset(
        PATH, NAME,
        # transform=ToTemporalData(),  # if necessary.
        **_dataset_kwargs,
    )
    if isinstance(_dataset, tuple):
        for _d in _dataset:
            print(_d[0])
        print("-" * 10)
        _all_dataset = _dataset
        _dataset = _dataset[0]
    else:
        _all_dataset = None
    print("Using", _dataset)
    for d in _dataset:
        print(d)

    _data = _dataset[0]

    if NAME.startswith("Singleton"):
        print("t", torch.unique(_data.t))
        _split = _dataset.get_rel_split()
        _train, _val, _test = _all_dataset
        print("split", _split)
        print(
            "split_index",
            _val[0].edge_index[:, _split["val"]].size(),
            _test[0].edge_index[:, _split["test"]].size(),
        )

    if NAME.startswith("JODIEDataset"):
        from collections import Counter

        print(torch.unique(_data.t))
        for i, (k, v) in enumerate(Counter(_data.t.tolist()).most_common()):
            print(k, v)
            if i > 20:
                break

    if NAME == "ogbn-arxiv":
        _split_idx = _dataset.get_idx_split()
        print(_split_idx["train"])  # tensor([     0,      1,      2,  ..., 169145, 169148, 169251])
        print(
            _split_idx["train"].size(),
            _split_idx["valid"].size(),
            _split_idx["test"].size(),
        )  # torch.Size([90941]) torch.Size([29799]) torch.Size([48603])

    if NAME == "ogbl-collab" and isinstance(_data, Data):
        _split_edge = _dataset.get_edge_split()
        _pos_train_edge = _split_edge["train"]["edge"]
        _pos_train_year = _split_edge["train"]["year"]
        _pos_valid_edge = _split_edge["valid"]["edge"]
        _neg_valid_edge = _split_edge["valid"]["edge_neg"]
        _pos_test_edge = _split_edge["test"]["edge"]
        _neg_test_edge = _split_edge["test"]["edge_neg"]
        assert _pos_train_edge.size() == torch.Size([1179052, 2])
        assert _pos_train_year.size() == torch.Size([1179052])
        assert _pos_valid_edge.size() == torch.Size([60084, 2])
        assert _neg_valid_edge.size() == torch.Size([100000, 2])
        assert torch.unique(_pos_train_edge).size() == torch.unique(_data.edge_index).size()
        assert torch.allclose(_pos_train_edge.float().std(), _dataset[0].edge_index.float().std())
