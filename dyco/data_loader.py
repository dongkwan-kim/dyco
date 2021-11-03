import os
from itertools import zip_longest
from pprint import pprint
from typing import List, Dict, Union, Tuple, Callable, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Data, Batch, TemporalData

from termcolor import cprint
from torch_geometric.loader.dataloader import Collater
from torch_geometric.utils import add_self_loops
from tqdm import tqdm

from dataset import get_dynamic_graph_dataset
from data_utils import Loading, CoarseSnapshotData, from_temporal_to_singleton_data, add_trivial_neg_edges
from utils import (torch_setdiff1d, to_index_chunks_by_values,
                   subgraph_and_edge_mask, exist_attr, startswith_any, idx_to_mask, iter_transform, rename_attr,
                   func_compose)


def _add_trivial_negatives_to_snapshot(data: CoarseSnapshotData) -> CoarseSnapshotData:
    assert hasattr(data, "train_edge_index")
    data.train_edge_index, data.train_neg_edge_index = add_trivial_neg_edges(
        data.train_edge_index, data.num_nodes)
    return data


def _rename_to_pos_and_neg_edge(data: Union[Batch, Data]) -> Batch:
    assert hasattr(data, "train_edge_index")
    assert hasattr(data, "train_neg_edge_index")
    data.pos_edge = data.train_edge_index
    data.neg_edge = data.train_neg_edge_index
    data.__delattr__("train_edge_index")
    data.__delattr__("train_neg_edge_index")
    return data


class SnapshotGraphLoader(DataLoader):

    def __init__(self, data: Union[Data, TemporalData, Batch],
                 loading_type=Loading.coarse,
                 batch_size=1, step_size=1,
                 split_edge_of_last_snapshot=False,
                 shuffle=True, num_workers=0,
                 follow_batch=None, exclude_keys=None,
                 transform_after_collation=None,
                 snapshot_dir="./", num_nodes=None,
                 edge_split_idx=None, edge_batch_size=None,
                 **kwargs):

        self._data = data
        self.loading_type = loading_type
        self.step_size = step_size
        self.split_edge_of_last_snapshot = split_edge_of_last_snapshot

        self.shuffle = shuffle
        self.follow_batch = follow_batch or []
        self.exclude_keys = exclude_keys or []
        self.collater = Collater(follow_batch, exclude_keys)

        self.transform_after_collation: Callable[[Batch], Batch] = transform_after_collation
        self.snapshot_dir = snapshot_dir
        self.transform_per_snapshot: Optional[Callable[[CoarseSnapshotData], CoarseSnapshotData]] = None

        self.edge_batch_size = edge_batch_size
        if edge_split_idx is not None:  # for ogbl-collab
            data.train_edge_index = edge_split_idx["train"]["edge"].t()
            data.train_edge_year = edge_split_idx["train"]["year"]
            # Add train_edge_index, train_neg_edge_index
            self.transform_per_snapshot = _add_trivial_negatives_to_snapshot
            # Rename to pos_edge, neg_edge
            self.transform_after_collation = func_compose(
                self.transform_after_collation, _rename_to_pos_and_neg_edge)

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
            split_edge = dataset.get_edge_split()
        except AttributeError:
            split_edge = None
        return dict(snapshot_dir=dataset.processed_dir,
                    num_nodes=num_nodes,
                    edge_split_idx=split_edge)

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

    def __len__(self):
        # Cannot pre-compute __len__ if you use edge_batch_size.
        return super().__len__() if self.edge_batch_size is None else None

    def __iter__(self):
        __iter__ = super().__iter__()
        if self.edge_batch_size is None:
            raise NotImplementedError(
                "Something wrong in negative sampling,"
                "use edge_batch_size for now. Will I fix it? IDK")
        else:
            for idx, elem in enumerate(__iter__):
                edge_loader = EdgeLoader(
                    batch_size=self.edge_batch_size,
                    pos_edge_index=elem.pos_edge.t(),
                    neg_edge_index="trivial_random_samples",
                    shuffle=self.shuffle,
                    num_nodes=elem.x.size(0),
                )
                for edge_batch in edge_loader:
                    elem.pos_edge = edge_batch.pos_edge
                    elem.neg_edge = edge_batch.neg_edge
                    yield elem

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
            b = CoarseSnapshotData.to_batch(
                data_at_t_list,
                pernode_attrs=self.attr_requirements,
                transform_per_snapshot=self.transform_per_snapshot,
                transform_per_batch=self.transform_after_collation)
        else:
            raise ValueError
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

        if not exist_attr(data, "edge_index"):
            raise Exception("Please check that whether the data has edge_index attribute.")

        time_step = getattr(data, time_name)  # e.g., node_year, edge_year, t

        index_chunks_dict = to_index_chunks_by_values(time_step)  # Dict[Any, LongTensor]:
        train_index_chunks_dict = to_index_chunks_by_values(
            getattr(data, "train_edge_year")) if exist_attr(data, "train_edge_year") else None

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

                if torch_setdiff1d(indices, torch.unique(sub_edge_index)).size(0) != 0:
                    raise Exception("Use add_self_loops to edge_index to handle isolated nodes")

            # index chunks are edges.
            # e.g., Data(edge_index=[2, E], edge_weight=[E, 1], edge_year=[E, 1], x=[E, F])
            elif "edge" in time_name:
                sub_edge_index = remained_edge_index[:, indices]
                if train_index_chunks_dict is not None:
                    train_indices = train_index_chunks_dict[curr_time]
                    kwg_for_data["train_edge_index"] = data.train_edge_index[:, train_indices]

            # index chunks are relations or events.
            # e.g., Data(edge_index=[2, 373018], rel=[373018, 1], t=[373018, 1])
            elif time_name == "t":
                sub_edge_index = remained_edge_index[:, indices]
                for attr_name in ["rel", "edge_attr", "edge_y"]:
                    if exist_attr(data, attr_name):
                        o = getattr(data, attr_name)
                        kwg_for_data[attr_name] = o[indices]
            else:
                raise ValueError(f"Wrong time_name: {time_name}")

            t = torch.Tensor([curr_time])
            data_at_curr = CoarseSnapshotData(t=t, edge_index=sub_edge_index, **kwg_for_data)
            data_list.append(data_at_curr)

        if save_cache:
            cprint(f"Save snapshots at {self.snapshot_path}", "green")
            torch.save(data_list, self.snapshot_path)

        return data_list


class EdgeLoader:

    def __init__(self, batch_size, pos_edge_index, neg_edge_index=None,
                 num_nodes=None, additional_kwargs: Dict = None,
                 batch_idx_to_add_kwargs: Union[str, int] = "all",
                 shuffle=False, **kwargs):
        self.num_nodes = num_nodes  # for _add_trivial_negatives
        self.additional_kwargs = additional_kwargs or {}
        self.batch_idx_to_add_kwargs = batch_idx_to_add_kwargs
        assert batch_idx_to_add_kwargs == "all" or isinstance(batch_idx_to_add_kwargs, int)
        self.pos_loader = DataLoader(pos_edge_index, batch_size, shuffle, collate_fn=self.__collate__, **kwargs)

        assert (neg_edge_index is None) or \
               (neg_edge_index == "trivial_random_samples") or \
               (isinstance(neg_edge_index, torch.Tensor))
        self.neg_loader, self.use_neg_transform = None, False
        if neg_edge_index is not None and isinstance(neg_edge_index, torch.Tensor):
            self.neg_loader = DataLoader(neg_edge_index, batch_size, shuffle, collate_fn=self.__collate__, **kwargs)
        elif neg_edge_index == "trivial_random_samples":
            self.use_neg_transform = True

    @staticmethod
    def __collate__(index_list):
        return default_collate(index_list).t()  # transpose to make [2, E]

    def get_add_trivial_negatives(self) -> Callable:

        def _add_trivial_negatives(pos_edge_index) -> Tuple[Tensor, Tensor]:
            return add_trivial_neg_edges(pos_edge_index, self.num_nodes)

        # pos_edge_index, neg_edge_index
        return _add_trivial_negatives

    def format_iteration(self, enumerated_edge_pair):
        idx, (pos_edge, neg_edge) = enumerated_edge_pair
        batch_dict = dict(pos_edge=pos_edge, neg_edge=neg_edge)
        if self.batch_idx_to_add_kwargs == "all":
            batch_dict.update(**self.additional_kwargs)
        elif self.batch_idx_to_add_kwargs == idx:
            batch_dict.update(**self.additional_kwargs)
        return Data(**batch_dict)

    def __iter__(self):

        if self.neg_loader is None and not self.use_neg_transform:
            it = zip_longest(iter(self.pos_loader), [], fillvalue=None)
        elif self.neg_loader is not None and isinstance(self.neg_loader, DataLoader):
            it = zip_longest(self.pos_loader, self.neg_loader, fillvalue=None)
        elif self.use_neg_transform:
            it = iter_transform(self.pos_loader, transform=self.get_add_trivial_negatives())
        else:
            raise AttributeError(f"neg_loader={self.neg_loader}, use_neg_transform={self.use_neg_transform}")
        return iter_transform(enumerate(it), transform=self.format_iteration)

    def __len__(self):
        if self.neg_loader is not None and isinstance(self.neg_loader, DataLoader):
            return max(len(self.pos_loader), len(self.neg_loader))
        else:
            return len(self.pos_loader)


if __name__ == "__main__":
    from pytorch_lightning import seed_everything

    seed_everything(43)

    PATH = "/mnt/nas2/GNN-DATA/PYG/"
    NAME = "ogbl-collab"
    LOADER = "SnapshotGraphLoader"
    # JODIEDataset/reddit, JODIEDataset/wikipedia, JODIEDataset/mooc, JODIEDataset/lastfm
    # ogbn-arxiv, ogbl-collab, ogbl-citation2
    # SingletonICEWS18, SingletonGDELT
    # BitcoinOTC
    _dataset_kwargs = {}
    if NAME.startswith("Singleton"):
        _dataset_kwargs["splits"] = ["train", "train_val", "train_val_test"]

    _dataset = get_dynamic_graph_dataset(PATH, NAME, **_dataset_kwargs)
    if isinstance(_dataset, tuple):
        _dataset = _dataset[0]

    _data = _dataset[0]
    if NAME == "ogbn-arxiv":
        _data.edge_index = add_self_loops(_data.edge_index, num_nodes=_data.x.size(0))[0]

    if LOADER == "SnapshotGraphLoader":
        _loader = SnapshotGraphLoader(
            _data,
            loading_type=SnapshotGraphLoader.get_loading_type(NAME),
            batch_size=3, step_size=1,
            **SnapshotGraphLoader.get_kwargs_from_dataset(_dataset),
        )
        cprint("SnapshotGraphLoader wo/ edge_batch_size", "green")
        for i, _batch in enumerate(tqdm(_loader)):
            # e.g., Batch(batch=[26709], edge_index=[2, 48866], ptr=[4], t=[3], x=[26709, 128], y=[26709, 1])
            if i < 2:
                print(f"ei {_batch.edge_index.min().item()} -- {_batch.edge_index.max().item()}", end=" / ")
                try:
                    print(f"pe {_batch.pos_edge.min().item()} -- {_batch.pos_edge.max().item()}",
                          end=" / ")
                except AttributeError:
                    pass
                print("t =", _batch.t, end=" / ")
                cprint(_batch, "yellow")
            else:
                break

        _loader = SnapshotGraphLoader(
            _data,
            loading_type=SnapshotGraphLoader.get_loading_type(NAME),
            batch_size=55, step_size=1,
            edge_batch_size=65536,
            **SnapshotGraphLoader.get_kwargs_from_dataset(_dataset),
        )
        cprint("SnapshotGraphLoader valid w/ edge_batch_size", "green")
        for i, _batch in enumerate(tqdm(_loader)):
            cprint(_batch, "yellow")
            print("neg_edge", _batch.neg_edge.sum())

        cprint("Again, SnapshotGraphLoader valid w/ edge_batch_size", "green")
        for i, _batch in enumerate(tqdm(_loader)):
            cprint(_batch, "yellow")
            print("neg_edge", _batch.neg_edge.sum())

    elif LOADER == "EdgeLoader":
        assert NAME == "ogbl-collab"
        _split_edge = _dataset.get_edge_split()
        _pos_valid_edge = _split_edge['valid']['edge']
        _neg_valid_edge = _split_edge['valid']['edge_neg']

        cprint("-- w/ trivial_random_samples and kwargs_at_first_batch", "green")
        _loader = EdgeLoader(
            batch_size=8, pos_edge_index=_pos_valid_edge, neg_edge_index="trivial_random_samples",
            additional_kwargs={"wow": 123}, batch_idx_to_add_kwargs="all",
            num_nodes=_data.num_nodes)
        for i, _batch in enumerate(tqdm(_loader)):
            pprint(_batch)
            if i == 1:
                break

        cprint("-- w/ neg_edge_index", "green")
        _loader = EdgeLoader(
            batch_size=8, pos_edge_index=_pos_valid_edge, neg_edge_index=_neg_valid_edge,
            additional_kwargs={"wow": 123}, batch_idx_to_add_kwargs=0,
            num_nodes=_data.num_nodes)
        for i, _batch in enumerate(tqdm(_loader)):
            pprint(_batch)
            if i == 1:
                break

        cprint("-- wo/ neg_edge_index", "green")
        _loader = EdgeLoader(
            batch_size=8, pos_edge_index=_pos_valid_edge, neg_edge_index=None,
            num_nodes=_data.num_nodes)
        for i, _batch in enumerate(tqdm(_loader)):
            pprint(_batch)
            if i == 1:
                break
