import os
from pprint import pprint
from typing import List, Dict, Union

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch, TemporalData

from termcolor import cprint
from torch_geometric.data.dataloader import Collater
from torch_geometric.utils import add_self_loops
from tqdm import tqdm

from dataset import get_dynamic_graph_dataset
from data_utils import Loading, CoarseSnapshotData, from_temporal_to_singleton_data
from utils import (torch_setdiff1d, to_index_chunks_by_values,
                   subgraph_and_edge_mask, exist_attr, startswith_any, idx_to_mask)


class SnapshotGraphLoader(DataLoader):

    def __init__(self, data: Union[Data, TemporalData, Batch],
                 loading_type=Loading.coarse,
                 batch_size=1, step_size=1,
                 shuffle=True, num_workers=0,
                 follow_batch=None, exclude_keys=None,
                 transform=None,
                 snapshot_dir="./", num_nodes=None, node_split_idx=None,
                 **kwargs):

        self._data = data
        self.loading_type = loading_type
        self.step_size = step_size

        self.follow_batch = follow_batch or []
        self.exclude_keys = exclude_keys or []
        self.collater = Collater(follow_batch, exclude_keys)

        self.transform = transform
        self.snapshot_dir = snapshot_dir

        node_split_mask = idx_to_mask(node_split_idx, num_nodes)  # for ogbn-arxiv
        data.train_mask = node_split_mask["train"]

        if self.loading_type == Loading.coarse:
            # e.g., ogbn, ogbl, singleton*
            if isinstance(data, TemporalData):
                data: Data = from_temporal_to_singleton_data(data)
            for attr_name in ["node_year", "edge_year", "t"]:
                if exist_attr(data, attr_name):
                    time_name = attr_name
                    break
            else:
                raise AttributeError("No node_year or edge_year or t")

            # A complete snapshot at t is a cumulation of data_list from 0 -- t-1.
            self.snapshot_list: List[CoarseSnapshotData] = self.disassemble_to_multi_snapshots(
                data, time_name, save_cache=True)
            self.num_snapshots = len(self.snapshot_list)
            self.attr_requirements = {k: getattr(data, k) for k in data.keys if k in ["x", "y", "train_mask"]}
            if "x" not in self.attr_requirements:
                # Add x as indices.
                assert num_nodes is not None
                self.attr_requirements["x"] = torch.arange(num_nodes).view(-1, 1)

        elif self.loading_type == Loading.fine:
            raise NotImplementedError

        else:  # others:
            raise ValueError("Wrong loading type: {}".format(self.loading_type))

        # indices are sampled from [step_size - 1, step_size, ..., len(snapshots) - 1]
        super(SnapshotGraphLoader, self).__init__(
            range(self.step_size - 1, self.num_snapshots),
            batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
            collate_fn=self.__collate__, **kwargs,
        )

    @staticmethod
    def get_kwargs_from_dataset(dataset) -> dict:
        try:
            num_nodes = dataset.num_nodes
        except AttributeError:
            num_nodes = None
        try:
            split_idx = dataset.get_idx_split()
        except AttributeError:
            split_idx = None
        return dict(snapshot_dir=dataset.processed_dir,
                    num_nodes=num_nodes,
                    node_split_idx=split_idx)

    @staticmethod
    def get_loading_type(dataset_name: str) -> Loading:
        if startswith_any(dataset_name, ["ogb", "Singleton", "JODIEDataset"]):
            return Loading.coarse
        else:
            raise ValueError("Wrong name: {}".format(dataset_name))

    @property
    def B(self):
        return self.batch_size

    @property
    def S(self):
        return self.step_size

    @property
    def snapshot_path(self):
        return os.path.join(self.snapshot_dir, "snapshots.pt")

    def __collate__(self, index_list) -> Batch:
        # Construct (low, high) indices per batch
        indices_high = torch.Tensor(index_list).long() + 1
        indices_low = indices_high - self.S
        indices_ranges = torch.stack([indices_low, indices_high]).t().tolist()

        # Convert indices to snapshots, then collate to single Batch.
        data_at_t_list = []
        if self.loading_type == Loading.coarse:
            for low, high in indices_ranges:
                snapshot_sublist = self.snapshot_list[low:high]
                data_at_t_within_steps = CoarseSnapshotData.aggregate(snapshot_sublist)
                data_at_t_list.append(data_at_t_within_steps)
            b = CoarseSnapshotData.to_batch(data_at_t_list, self.attr_requirements)
        else:
            raise ValueError
        b = self.transform(b) if self.transform is not None else b
        return b

    def disassemble_to_multi_snapshots(self, data: Data, time_name: str,
                                       save_cache: bool = True) -> List[CoarseSnapshotData]:
        if save_cache:
            try:
                snapshots = torch.load(self.snapshot_path)
                cprint(f"Load snapshots from {self.snapshot_path}", "green")
                return snapshots
            except FileNotFoundError:
                pass
            except Exception as e:
                cprint(f"Load snapshots failed from {self.snapshot_path}, the error is {e}", "red")

        time_step = getattr(data, time_name)  # e.g., node_year, edge_year, t

        index_chunks_dict = to_index_chunks_by_values(time_step)  # Dict[Any, LongTensor]:

        remained_edge_index = data.edge_index.clone()
        indices_until_curr = None
        num_nodes = data.x.size(0) if exist_attr(data, "x") else None
        data_list, kwg_for_data = [], {}
        for curr_time, indices in sorted(index_chunks_dict.items()):

            # index_chunks are nodes.
            if "node" in time_name:
                indices_until_curr = indices if indices_until_curr is None else \
                    torch.cat([indices_until_curr, indices])
                # edge_index construction
                sub_edge_index, _, edge_mask = subgraph_and_edge_mask(
                    indices_until_curr, remained_edge_index,
                    relabel_nodes=False, num_nodes=num_nodes)
                remained_edge_index = remained_edge_index[:, ~edge_mask]  # edges not in sub_edge_index
                # x_index_all set-minus x_index_in_edges
                iso_x_index = torch_setdiff1d(indices, torch.unique(sub_edge_index))

            # index chunks are edges.
            # e.g., Data(edge_index=[2, E], edge_weight=[E, 1], edge_year=[E, 1], x=[E, F])
            elif "edge" in time_name:
                sub_edge_index = remained_edge_index[:, indices]
                iso_x_index = None

            # index chunks are relations or events.
            # e.g., Data(edge_index=[2, 373018], rel=[373018, 1], t=[373018, 1])
            elif time_name == "t":
                sub_edge_index = remained_edge_index[:, indices]
                iso_x_index = None
                for attr_name in ["rel", "edge_attr", "edge_y"]:
                    if exist_attr(data, attr_name):
                        o = getattr(data, attr_name)
                        kwg_for_data[attr_name] = o[indices]
            else:
                raise ValueError(f"Wrong time_name: {time_name}")

            t = torch.Tensor([curr_time])
            data_at_curr = CoarseSnapshotData(
                t=t, edge_index=sub_edge_index, iso_x_index=iso_x_index,
                **kwg_for_data)
            data_list.append(data_at_curr)

        if save_cache:
            cprint(f"Save snapshots at {self.snapshot_path}", "green")
            torch.save(data_list, self.snapshot_path)

        return data_list


def get_snapshot_graph_loader(dataset, name: str, stage: str, *args, **kwargs):
    assert stage in ["train", "valid", "test", "evaluation"]
    loader = SnapshotGraphLoader(
        dataset, loading_type=SnapshotGraphLoader.get_loading_type(name),
        *args, **kwargs,
    )
    return loader


if __name__ == "__main__":
    from pytorch_lightning import seed_everything

    seed_everything(43)

    PATH = "/mnt/nas2/GNN-DATA/PYG/"
    NAME = "ogbn-arxiv"
    # JODIEDataset/reddit, JODIEDataset/wikipedia, JODIEDataset/mooc, JODIEDataset/lastfm
    # ogbn-arxiv, ogbl-collab, ogbl-citation2
    # SingletonICEWS18, SingletonGDELT
    # BitcoinOTC

    _dataset = get_dynamic_graph_dataset(PATH, NAME)
    if isinstance(_dataset, tuple):
        _dataset = _dataset[0]

    _loader = SnapshotGraphLoader(
        _dataset[0],
        loading_type=SnapshotGraphLoader.get_loading_type(NAME),
        batch_size=3, step_size=4,
        **SnapshotGraphLoader.get_kwargs_from_dataset(_dataset),
    )
    for i, _batch in enumerate(tqdm(_loader)):
        # e.g.,
        # Batch(batch=[26709], edge_index=[2, 48866], iso_x_index=[1747], iso_x_index_batch=[1747],
        #       ptr=[4], t=[3], x=[26709, 128], y=[26709, 1])
        if i < 2:
            print("\n t =", _batch.t, end=" / ")
            cprint(_batch, "yellow")
        else:
            exit()
