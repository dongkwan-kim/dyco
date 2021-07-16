from torch_geometric.data import Batch, Data


def to_singleton_data(_data_list):
    b = Batch.from_data_list(_data_list)
    return Data(**{k: getattr(b, k) for k in b.keys})
