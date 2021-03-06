from enum import Enum, auto
from pprint import pprint
from typing import List, Dict, Union, Tuple, Callable

import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import Tensor
from torch_geometric.data import Batch, Data, TemporalData
from torch_geometric.transforms import ToSparseTensor, ToUndirected, Compose
from torch_geometric.typing import OptTensor
from torch_geometric.utils import to_undirected, add_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor

from utils import exist_attr, torch_setdiff1d, ld_to_dl

BatchType = Union[Batch, Data]


class Loading(Enum):
    coarse = auto()
    fine = auto()


class CoarseSnapshotData(Data):
    """CoarseSnapshotData has multiple nodes or edges per given time stamp"""

    def __init__(self, x: OptTensor = None, edge_index: OptTensor = None,
                 edge_attr: OptTensor = None, y: OptTensor = None, pos: OptTensor = None,
                 **kwargs):
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        # * Very IMPORTANT *
        # original code: self.num_nodes if bool(re.search("(index|face)", key)) else 0
        return 0

    def is_num_nodes_inferred_by_edge_index(self):
        for k in ["__num_nodes__", "x", "pos", "normal", "batch", "adj", "adj_t", "face"]:
            if exist_attr(self, k):
                return False
        else:
            return True

    @property
    def num_nodes(self):
        if self.is_num_nodes_inferred_by_edge_index():  # remove warnings.
            return maybe_num_nodes(self.edge_index)
        else:
            return super(CoarseSnapshotData, self).num_nodes

    @classmethod
    def aggregate(cls, snapshot_sublist, t_position=-1, follow_batch=None, exclude_keys=None):
        """
        :return: e.g., SnapshotData(edge_index=[2, E], t=[1])
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
        return data_aggr_at_t

    @staticmethod
    def to_batch(snapshot_sublist: List,
                 pernode_attrs: Dict[str, Tensor] = None,
                 num_nodes: int = None,
                 transform_per_snapshot: Callable = None,
                 transform_per_batch: Callable = None) -> BatchType:
        """
        :param snapshot_sublist: List[SnapshotData],
            e.g., SnapshotData(edge_index=[2, E], t=[1])
        :param pernode_attrs: Dict[str, Tensor]
            e.g., {"x": tensor([[...], [...]]), "y": tensor([[...], [...]])}
        :param num_nodes: if pernode_attrs is not given, use this.
        :param transform_per_snapshot: Transform from CoarseSnapshotData to CoarseSnapshotData
        :param transform_per_batch: Transform from Batch to Batch
        :return: Batch,
            e.g., Batch(batch=[26709], edge_index=[2, 48866],
                        ptr=[4], t=[3], x=[26709, 128], y=[26709, 1])
        """
        assert isinstance(snapshot_sublist[0], CoarseSnapshotData)
        snapshot_sublist: List[CoarseSnapshotData] = sorted(snapshot_sublist, key=lambda d: d.t)
        pernode_attrs = pernode_attrs or dict()
        if "x" in pernode_attrs:
            num_nodes = pernode_attrs["x"].size(0)

        mask = torch.zeros(num_nodes, dtype=torch.bool)
        assoc = torch.full((num_nodes,), -1, dtype=torch.long)

        # Construct assoc with an increasing order by time: 1971, ..., 2020
        start = 0
        for b_no, data in enumerate(snapshot_sublist):
            existing_nodes = data.edge_index.view(-1)
            mask[existing_nodes] = 1
            mask[assoc >= 0] = 0  # Unmask pre-allocated nodes by previous events.
            num_added_nodes = mask.sum().item()

            # Masked nodes start with the accumulated number of nodes until T,
            # and the length will be the number of newly added nodes.
            assoc[mask] = torch.arange(start, start + num_added_nodes)
            start += num_added_nodes

            # Distribute pernode attributes, such as x, y, etc.
            for attr_name, pernode_tensor in pernode_attrs.items():
                setattr(data, attr_name, pernode_tensor[mask])

            # Re-init mask only, not assoc
            mask[:] = 0

        # Relabel *_index
        data_list = []
        for data in snapshot_sublist:
            for k in data.keys:
                if "index" in k:  # *_edge_index
                    # e.g., data.edge_index = assoc[data.edge_index]
                    setattr(data, k, assoc[getattr(data, k)])

            # Transform data if necessary
            data if transform_per_snapshot is None else transform_per_snapshot(data)
            data_list.append(data)

        b = Batch.from_data_list(data_list)

        return b if transform_per_batch is None else transform_per_batch(b)


def to_singleton_data(data_list) -> Data:
    b = Batch.from_data_list(data_list)
    return Data(**{k: getattr(b, k) for k in b.keys})


def from_events_to_singleton_data(data_list: List[Data]) -> Data:
    """Convert (subject entity, relation, object, entity, time)
        e.g., Data(obj=[373018], rel=[373018], sub=[373018], t=[373018])
        --> Data(edge_index=[sub, obj], ...)
    :param data_list: List[Data]
    :return: Data,
        e.g., Data(edge_index=[2, 373018], rel=[373018, 1], t=[373018, 1])
    """
    s_data = to_singleton_data(data_list)
    edge_index = torch.stack([s_data.sub, s_data.obj])
    rel, t = s_data.rel.view(-1, 1), s_data.t.view(-1, 1)
    return Data(edge_index=edge_index, rel=rel, t=t)


def from_temporal_to_singleton_data(data_or_data_list: Union[TemporalData, List[TemporalData]]) -> Data:
    """Convert (dst, src, msg, t, y)
        e.g., TemporalData(dst=[E], msg=[E, F], src=[E], t=[E], y=[E])
        --> Data(edge_index=[src, dst], ...)
    :param data_or_data_list: List[TemporalData] or TemporalData
    :return: Data,
        e.g., Data(edge_attr=[157474, 172], edge_index=[2, 157474], t=[157474, 1], y=[157474, 1])
    """
    t_data = data_or_data_list if isinstance(data_or_data_list, TemporalData) else data_or_data_list[0]
    edge_index = torch.stack([t_data.src, t_data.dst])
    t, edge_y = t_data.t.view(-1, 1), t_data.y.view(-1, 1)
    return Data(edge_index=edge_index, edge_attr=t_data.msg, t=t, edge_y=edge_y)


def from_events_to_temporal_data(data_list: List[Data]) -> TemporalData:
    """Convert (subject entity, relation, object, entity, time)
        e.g., Data(obj=[373018], rel=[373018], sub=[373018], t=[373018])
        --> TemporalData
    :param data_list: List[Data]
    :return: TemporalData,
        e.g., TemporalData(dst=[373018], src=[373018], t=[373018], y=[373018])
    """
    s_data = to_singleton_data(data_list)
    return TemporalData(src=s_data.sub, dst=s_data.obj, t=s_data.t, y=s_data.rel)


def from_singleton_to_temporal_data(data_or_data_list: Union[Data, List[Data]]) -> TemporalData:
    """Convert (edge_index=[sub, obj], ...)
        e.g., Data(edge_index=[2, 373018], rel=[373018, 1], t=[373018, 1])
        --> Data(src=sub, dst=obj)
    :param data_or_data_list: List[Data] or Data
    :return: TemporalData,
        e.g., TemporalData(dst=[373018], src=[373018], t=[373018], y=[373018])
    """
    data = data_or_data_list if isinstance(data_or_data_list, Data) else data_or_data_list[0]
    sub, obj = data.edge_index
    return TemporalData(src=sub, dst=obj, t=data.t.squeeze(), y=data.rel.squeeze())


def from_ogb_data_to_temporal_data(data_or_data_list: Union[Data, List[Data]]) -> TemporalData:
    """Convert Data with node_year or edge_year
        e.g., Data(edge_index=[2, 1166243], node_year=[169343, 1], x=[169343, 128], y=[169343, 1])
              Data(edge_index=[2, 2358104], edge_weight=[2358104, 1], edge_year=[2358104, 1], x=[235868, 128])
              Data(edge_index=[2, 30387995], node_year=[2927963, 1], x=[2927963, 128])
        --> TemporalData
    :param data_or_data_list: List[Data] or Data
    :return: TemporalData,
        e.g., TemporalData(dst=[E], src=[E], t=[E], x=[N, F])
    """
    data = data_or_data_list if isinstance(data_or_data_list, Data) else data_or_data_list[0]
    if hasattr(data, "node_year"):
        # Construct event per edge using node_year
        #  e.g., edge of (2015, 2017) --> event at 2017.
        node_year = data.node_year.squeeze()
        node_pair_year = node_year[data.edge_index]  # [2, E]
        t, perm = torch.sort(torch.max(node_pair_year, dim=0).values)  # [E]
        src, dst = data.edge_index[:, perm]
        kwargs = {"x": data.x, "y": data.y.squeeze()}
    elif hasattr(data, "edge_year"):
        t, perm = torch.sort(data.edge_year.squeeze())
        src, dst = data.edge_index[:, perm]
        kwargs = {"x": data.x, "edge_weight": data.edge_weight[perm].squeeze()}
    else:
        raise AttributeError
    return TemporalData(src=src, dst=dst, t=t, **kwargs)


class ToTemporalData(object):

    def __call__(self, data):
        if isinstance(data, TemporalData):  # already TemporalData
            return data
        elif exist_attr(data, "node_year") or exist_attr(data, "edge_year"):  # ogb
            return from_ogb_data_to_temporal_data(data)
        elif exist_attr(data, "rel") and data.rel.numel() > 1:  # tkg
            return from_singleton_to_temporal_data(data)
        else:
            raise ValueError

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class FromTemporalData(object):

    def __call__(self, data):
        if isinstance(data, Data):  # already data
            return data
        elif isinstance(data, TemporalData):
            return from_temporal_to_singleton_data(data)
        else:
            raise TypeError("Wrong type: {}".format(type(data)))

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class ToSymSparseTensor(ToSparseTensor):

    def __call__(self, data):
        data = super().__call__(data)
        data.adj_t = data.adj_t.to_symmetric()
        return data


class ToSymmetric(object):

    def __init__(self, to_sparse_tensor=False, *args, **kwargs):
        if to_sparse_tensor:
            self.transform = ToSymSparseTensor(*args, **kwargs)
        else:
            self.transform = ToUndirected(*args, **kwargs)

    def __call__(self, data):
        return self.transform(data)


class UseValEdgesAsInput(object):
    """Implement https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/collab/gnn.py#L222-L236"""

    def __init__(self, to_sparse_tensor=False, to_symmetric=False, del_edge_weight=True):
        self.to_sparse_tensor = to_sparse_tensor
        self.to_symmetric = to_symmetric
        self.del_edge_weight = del_edge_weight
        self._val_edge_index = None

    def set_val_edge_index(self, split_idx):
        self._val_edge_index = split_idx['valid']['edge'].t()

    @classmethod
    def set_val_edge_index_for_compose(cls, compose_transform: Compose, split_idx):
        for t in compose_transform.transforms:
            if isinstance(t, cls):
                t.set_val_edge_index(split_idx)
                break

    def __call__(self, data):
        assert self._val_edge_index is not None
        edge_index = data.edge_index
        if not self.del_edge_weight:
            data.edge_weight = data.edge_weight.view(-1).to(torch.float)
        else:
            del data.edge_weight

        full_edge_index = torch.cat([edge_index, self._val_edge_index], dim=-1)
        if self.to_sparse_tensor:
            data = ToSparseTensor()(data)
            data.full_adj_t = SparseTensor.from_edge_index(full_edge_index).t()
            if self.to_symmetric:
                data.full_adj_t = data.full_adj_t.to_symmetric()
        else:
            if self.to_symmetric:
                full_edge_index = to_undirected(full_edge_index, num_nodes=data.x.size(0))
            data.full_edge_index = full_edge_index
        return data


def add_trivial_neg_edges(pos_edge_index, num_nodes) -> Tuple[Tensor, Tensor]:
    # Implement
    # https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/collab/gnn.py#L114
    neg_edge_index = torch.randint(0, num_nodes, pos_edge_index.size(), dtype=torch.long)
    return pos_edge_index, neg_edge_index


def replace_y_pred_neg_from_train_to_valid(model, prefix, outputs: EPOCH_OUTPUT) -> EPOCH_OUTPUT:
    if prefix == "valid":  # valid first
        model._cached_outputs = outputs
        return outputs

    elif prefix == "train":  # then train
        valid_outputs_as_dict = ld_to_dl(model._cached_outputs)
        model._cached_outputs = None

        train_outputs_as_dict = ld_to_dl(outputs)
        train_outputs_as_dict["y_pred_neg"] = valid_outputs_as_dict["y_pred_neg"]

        model.epoch_end("train", train_outputs_as_dict)
        return outputs  # for valid

    else:
        raise ValueError("Wrong prefix: {}".format(prefix))
