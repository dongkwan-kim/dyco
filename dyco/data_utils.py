from typing import List

import torch
from torch import Tensor
from torch_geometric.data import Batch, Data, TemporalData


def to_singleton_data(data_list) -> Data:
    b = Batch.from_data_list(data_list)
    return Data(**{k: getattr(b, k) for k in b.keys})


def from_events_to_singleton_data(data_list: List[Data]) -> Data:
    """Convert (subject entity, relation, object, entity, time)
        e.g., Data(obj=[373018], rel=[373018], sub=[373018], t=[373018]
        --> (edge_index=[sub, obj], ...)
    :param data_list: List[Data]
    :return: Data,
        e.g., Data(edge_index=[2, 373018], rel=[373018, 1], t=[373018, 1])
    """
    s_data = to_singleton_data(data_list)
    edge_index = torch.stack([s_data.sub, s_data.obj])
    rel, t = s_data.rel.view(-1, 1), s_data.t.view(-1, 1)
    return Data(edge_index=edge_index, rel=rel, t=t)
