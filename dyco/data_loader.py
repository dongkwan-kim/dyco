import os
from pprint import pprint
from typing import List, Dict, Union

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch

from termcolor import cprint
from torch_geometric.data.dataloader import Collater

from data import get_dynamic_graph_dataset
from data_utils import Loading, CoarseSnapshotData
from utils import torch_setdiff1d, to_index_chunks_by_values, subgraph_and_edge_mask, exist_attr


class SnapshotGraphLoader(DataLoader):

    def __init__(self, dataset, loading_type,
                 batch_size=1, step_size=1,
                 shuffle=True, num_workers=0,
                 follow_batch=None, exclude_keys=None,
                 **kwargs):

        self._dataset = dataset
        self.loading_type = loading_type
        self.step_size = step_size

        self.follow_batch = follow_batch or []
        self.exclude_keys = exclude_keys or []
        self.collater = Collater(follow_batch, exclude_keys)

        if self.loading_type == Loading.coarse:
            # e.g., ogbn, ogbl, singleton*
            data: Data = dataset[0]
            assert len(dataset) == 1
            for attr_name in ["node_year", "edge_year", "t"]:
                if exist_attr(data, attr_name):
                    time_name = attr_name
                    break
            else:
                raise AttributeError("No node_year or edge_year or t")

            # A complete snapshot at t is a cumulation of data_list from 0 -- t-1.
            self.snapshot_list: List[CoarseSnapshotData] = self.disassemble_to_multi_snapshots(
                dataset[0], time_name, save_cache=True)
            self.num_snapshots = len(self.snapshot_list)
            self.attr_requirements = {k: getattr(data, k) for k in data.keys if k in ["x", "y"]}
            if "x" not in self.attr_requirements:
                # Add x as indices.
                self.attr_requirements["x"] = torch.arange(self.num_nodes).view(-1, 1)

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

    @property
    def num_nodes(self):
        return self._dataset.num_nodes

    @property
    def B(self):
        return self.batch_size

    @property
    def S(self):
        return self.step_size

    @property
    def snapshot_path(self):
        return os.path.join(self._dataset.root, "snapshots.pt")

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
            return b
        else:
            raise ValueError

    @staticmethod
    def get_loading_type(dataset_name: str) -> Loading:
        if dataset_name.startswith("ogb") or dataset_name.startswith("Singleton"):
            return Loading.coarse
        else:
            raise ValueError("Wrong name: {}".format(dataset_name))

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
                if exist_attr(data, "rel"):
                    kwg_for_data["rel"] = data.rel[indices]

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
    NAME = "SingletonICEWS18"
    # JODIEDataset/reddit, JODIEDataset/wikipedia, JODIEDataset/mooc, JODIEDataset/lastfm
    # ogbn-arxiv, ogbl-collab, ogbl-citation2
    # SingletonICEWS18, SingletonGDELT
    # BitcoinOTC

    _dataset = get_dynamic_graph_dataset(PATH, NAME)
    _loader = get_snapshot_graph_loader(_dataset, NAME, "train", batch_size=3, step_size=4)
    for _batch in _loader:
        print(_batch)
