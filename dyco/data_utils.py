from enum import Enum, auto
from typing import List, Dict, Union

import torch
from torch import Tensor
from torch_geometric.data import Batch, Data, TemporalData
from torch_geometric.utils.num_nodes import maybe_num_nodes

from utils import exist_attr, torch_setdiff1d


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
    src, dst = data.edge_index
    t = (data.node_year if hasattr(data, "node_year") else data.edge_year).squeeze()
    kwargs = {k: getattr(data, k).squeeze() for k in data.keys
              if k not in ["edge_index", "node_year", "edge_year"]}
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
