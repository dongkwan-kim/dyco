import os
from pprint import pprint
from typing import List, Dict, Union
from enum import Enum, auto

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch

from termcolor import cprint
from torch_geometric.data.dataloader import Collater
from torch_geometric.utils.num_nodes import maybe_num_nodes

from data import get_dynamic_graph_dataset
from utils import torch_setdiff1d, to_index_chunks_by_values, subgraph_and_edge_mask, exist_attr


class Loading(Enum):
    coarse = auto()
    fine = auto()


class CoarseSnapshotData(Data):
    """CoarseSnapshotData has multiple nodes or edges per given time stamp"""
    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, pos=None, normal=None, face=None,
                 increase_num_nodes_for_index=None, **kwargs):
        self.increase_num_nodes_for_index = increase_num_nodes_for_index
        super().__init__(x, edge_index, edge_attr, y, pos, normal, face, **kwargs)

    def __inc__(self, key, value):
        if exist_attr(self, "increase_num_nodes_for_index") and self.increase_num_nodes_for_index:
            # original code: self.num_nodes if bool(re.search("(index|face)", key)) else 0
            return super(CoarseSnapshotData, self).__inc__(key, value)
        else:
            return 0

    def is_num_nodes_inferred_by_edge_index(self):
        for k in ["__num_nodes__", "x", "pos", "normal", "batch", "adj", "adj_t", "face"]:
            if exist_attr(self, k):
                return False
        else:
            return True

    @property
    def num_nodes(self):
        num_isolated_nodes = self.iso_x_index.size(0) if exist_attr(self, "iso_x_index") else 0
        if self.edge_index.numel() == 0:  # no edge graphs.
            return num_isolated_nodes
        elif self.is_num_nodes_inferred_by_edge_index():  # remove warnings.
            return maybe_num_nodes(self.edge_index) + num_isolated_nodes
        else:
            return super(CoarseSnapshotData, self).num_nodes

    @classmethod
    def aggregate(cls, snapshot_sublist, t_position=-1, follow_batch=None, exclude_keys=None):
        """
        :return: e.g., SnapshotData(edge_index=[2, E], iso_x_index=[S], t=[1])
        """
        assert isinstance(snapshot_sublist[0], CoarseSnapshotData)
        # t is excluded from the batch construction, since we only add t of given t_position.
        data_at_t = Batch.from_data_list(
            snapshot_sublist,
            follow_batch=follow_batch or [],
            exclude_keys=["t", *(exclude_keys or [])])
        data_at_t.t = snapshot_sublist[t_position].t

        data_aggr_at_t = cls(**{k: getattr(data_at_t, k) for k in data_at_t.keys
                                if k not in ["batch", "ptr"]})

        # There can be nodes that were isolated, but not any more after
        # concatenating more than two different snapshots.
        if len(snapshot_sublist) > 1 and snapshot_sublist[0].iso_x_index is not None:
            edge_index_until_t = torch.cat([s.edge_index for s in snapshot_sublist], dim=-1)
            # The last is excluded, since nodes do not travel across the time.
            iso_x_index_until_t_minus_1 = torch.cat([s.iso_x_index for s in snapshot_sublist[:-1]])
            # x_index_all set-minus x_index_in_edges
            data_aggr_at_t.iso_x_index = torch_setdiff1d(iso_x_index_until_t_minus_1,
                                                         edge_index_until_t)

        return data_aggr_at_t

    @staticmethod
    def to_batch(snapshot_sublist: List,
                 pernode_attrs: Dict[str, Tensor] = None,
                 num_nodes: int = None) -> Batch:
        """
        :param snapshot_sublist: List[SnapshotData],
            e.g., SnapshotData(edge_index=[2, E], iso_x_index=[N], t=[1])
        :param pernode_attrs: Dict[str, Tensor]
            e.g., {"x": tensor([[...], [...]]), "y": tensor([[...], [...]])}
        :param num_nodes: if pernode_attrs is not given, use this.
        :return: Batch,
            e.g., Batch(batch=[26709], edge_index=[2, 48866],
                        iso_x_index=[1747], iso_x_index_batch=[1747],
                        ptr=[4], t=[3], x=[26709, 128], y=[26709, 1])
        """
        assert isinstance(snapshot_sublist[0], CoarseSnapshotData)
        snapshot_sublist: List[CoarseSnapshotData]
        pernode_attrs = pernode_attrs or dict()
        if "x" in pernode_attrs:
            num_nodes = pernode_attrs["x"].size(0)

        # Relabel edge_index, iso_x_index (optional) of SnapshotData in
        # snapshot_sublist, and put pernode_attrs (e.g., x and y) to SnapshotData,
        # finally construct the Batch object with them.
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        assoc = torch.full((num_nodes,), -1, dtype=torch.long)
        for data in snapshot_sublist:

            if exist_attr(data, "iso_x_index"):
                existing_nodes = torch.cat([data.edge_index.view(-1), data.iso_x_index])
            else:
                existing_nodes = data.edge_index.view(-1)

            mask[existing_nodes] = 1
            assoc[mask] = torch.arange(mask.sum())
            data.edge_index = assoc[data.edge_index]

            if exist_attr(data, "iso_x_index"):
                data.iso_x_index = assoc[data.iso_x_index]

            # Distribute pernode attributes, such as x, y, etc.
            for attr_name, pernode_tensor in pernode_attrs.items():
                masked_tensor = pernode_tensor[mask]
                setattr(data, attr_name, masked_tensor)

            # *very* important for the batch construction
            data.increase_num_nodes_for_index = True

            # re-init mask and assoc
            mask[:] = 0
            assoc[:] = -1

        b = Batch.from_data_list(
            snapshot_sublist,
            follow_batch=["iso_x_index"], exclude_keys=["increase_num_nodes_for_index"],
        )
        return b


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
