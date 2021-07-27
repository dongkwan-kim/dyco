import os
from pprint import pprint
from typing import List, Dict

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch

from termcolor import cprint
from torch_geometric.data.dataloader import Collater
from torch_geometric.utils.num_nodes import maybe_num_nodes

from data import get_dynamic_graph_dataset
from utils import torch_setdiff1d, to_index_chunks_by_values, subgraph_and_edge_mask, exist_attr


class SnapshotData(Data):

    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, pos=None, normal=None, face=None,
                 increase_num_nodes_for_index=None, **kwargs):
        self.increase_num_nodes_for_index = increase_num_nodes_for_index
        super().__init__(x, edge_index, edge_attr, y, pos, normal, face, **kwargs)

    def __inc__(self, key, value):
        if exist_attr(self, "increase_num_nodes_for_index") and self.increase_num_nodes_for_index:
            # original code: self.num_nodes if bool(re.search("(index|face)", key)) else 0
            return super(SnapshotData, self).__inc__(key, value)
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
            return super(SnapshotData, self).num_nodes

    @classmethod
    def aggregate(cls, snapshot_sublist, t_position=-1, follow_batch=None, exclude_keys=None):
        """
        :return: e.g., SnapshotData(edge_index=[2, E], iso_x_index=[S], t=[1])
        """
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
        snapshot_sublist: List[SnapshotData]
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

        self.dataset = dataset
        self.loading_type = loading_type
        self.step_size = step_size

        self.follow_batch = follow_batch or []
        self.exclude_keys = exclude_keys or []
        self.collater = Collater(follow_batch, exclude_keys)

        if self.loading_type == "single-to-multi":  # e.g., ogbn, ogbl
            data: Data = dataset[0]
            assert len(dataset) == 1
            assert hasattr(data, "node_year") or hasattr(data, "edge_year"), "No node_year or edge_year"
            time_name = "node_year" if hasattr(data, "node_year") else "edge_year"
            # A complete snapshot at t is a cumulation of data_list from 0 -- t-1.
            self.snapshot_list = self.disassemble_to_multi_snapshots(dataset[0], time_name, save_cache=True)
            self.pernode_attrs = {k: getattr(data, k) for k in data.keys if k in ["x", "y"]}
        elif self.loading_type == "already-multi":
            raise NotImplementedError
        else:  # others:
            raise ValueError("Wrong loading type: {}".format(self.loading_type))

        # indices are sampled from [step_size - 1, step_size, ..., len(snapshots) - 1]
        super(SnapshotGraphLoader, self).__init__(
            range(self.step_size - 1, len(self.snapshot_list)),
            batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
            collate_fn=self.__collate__, **kwargs,
        )

    @property
    def B(self):
        return self.batch_size

    @property
    def S(self):
        return self.step_size

    @property
    def snapshot_path(self):
        return os.path.join(self.dataset.root, "snapshots.pt")

    def __collate__(self, index_list) -> Batch:
        # Construct (low, high) indices per batch
        indices_high = torch.Tensor(index_list).long() + 1
        indices_low = indices_high - self.S
        indices_ranges = torch.stack([indices_low, indices_high]).t().tolist()

        # Convert indices to snapshots, then collate to single Batch.
        data_at_t_list = []
        for low, high in indices_ranges:
            snapshot_sublist = self.snapshot_list[low:high]
            data_at_t_within_steps = SnapshotData.aggregate(snapshot_sublist)
            data_at_t_list.append(data_at_t_within_steps)

        b = SnapshotData.to_batch(data_at_t_list, self.pernode_attrs)
        return b

    @staticmethod
    def get_loading_type(dataset_name: str) -> str:
        if dataset_name.startswith("ogb"):
            return "single-to-multi"
        else:
            raise ValueError("Wrong name: {}".format(dataset_name))

    def disassemble_to_multi_snapshots(self, data: Data, time_name: str,
                                       save_cache: bool = True) -> List[SnapshotData]:
        if save_cache:
            try:
                snapshots = torch.load(self.snapshot_path)
                cprint(f"Load snapshots from {self.snapshot_path}", "green")
                return snapshots
            except FileNotFoundError:
                pass
            except Exception as e:
                cprint(f"Load snapshots failed from {self.snapshot_path}, the error is {e}", "red")

        time_step = getattr(data, time_name)  # e.g., node_year, edge_year

        index_chunks_dict = to_index_chunks_by_values(time_step)  # Dict[Any, LongTensor]:

        remained_edge_index = data.edge_index.clone()
        indices_until_curr = None
        num_nodes = data.x.size(0)
        data_list = []
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
            else:
                raise ValueError(f"Wrong time_name: {time_name}")

            t = torch.Tensor([curr_time])
            data_at_curr = SnapshotData(t=t, edge_index=sub_edge_index, iso_x_index=iso_x_index)
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
    # JODIEDataset/reddit, JODIEDataset/wikipedia, JODIEDataset/mooc, JODIEDataset/lastfm
    # ogbn-arxiv, ogbl-collab, ogbl-citation2
    # SingletonICEWS18, SingletonGDELT
    # BitcoinOTC
    NAME = "ogbn-arxiv"

    _dataset = get_dynamic_graph_dataset(PATH, NAME)
    _loader = get_snapshot_graph_loader(_dataset, NAME, "train", batch_size=3, step_size=4)
    for _batch in _loader:
        print(_batch)
