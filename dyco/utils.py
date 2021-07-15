from pprint import pprint
from typing import Dict, Any, List, Tuple

import torch
from torch import Tensor
from torch_geometric.utils import subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes


def to_index_chunks_by_values(tensor_1d: Tensor) -> Dict[Any, Tensor]:
    tensor_1d = tensor_1d.flatten()
    index_chunks_dict = dict()
    # todo: there can be more efficient way, but it might not be necessary.
    for v in torch.unique(tensor_1d):
        v = v.item()
        index_chunk = torch.nonzero(tensor_1d == v).flatten()
        index_chunks_dict[v] = index_chunk
    return index_chunks_dict


def subgraph_and_edge_mask(subset, edge_index, edge_attr=None, relabel_nodes=False,
                           num_nodes=None):
    """Same as the pyg.utils.subgraph except it returns edge_mask too."""
    device = edge_index.device

    if isinstance(subset, list) or isinstance(subset, tuple):
        subset = torch.tensor(subset, dtype=torch.long)

    if subset.dtype == torch.bool or subset.dtype == torch.uint8:
        n_mask = subset

        if relabel_nodes:
            n_idx = torch.zeros(n_mask.size(0), dtype=torch.long,
                                device=device)
            n_idx[subset] = torch.arange(subset.sum().item(), device=device)
    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        n_mask = torch.zeros(num_nodes, dtype=torch.bool)
        n_mask[subset] = 1

        if relabel_nodes:
            n_idx = torch.zeros(num_nodes, dtype=torch.long, device=device)
            n_idx[subset] = torch.arange(subset.size(0), device=device)

    mask = n_mask[edge_index[0]] & n_mask[edge_index[1]]
    edge_index = edge_index[:, mask]
    edge_attr = edge_attr[mask] if edge_attr is not None else None

    if relabel_nodes:
        edge_index = n_idx[edge_index]

    return edge_index, edge_attr, mask


if __name__ == '__main__':

    METHOD = "sort_and_relabel"

    if METHOD == "to_index_chunks_by_values":
        _tensor_1d = torch.Tensor([24, 20, 21, 21, 20, 23, 24])
        print(to_index_chunks_by_values(_tensor_1d))
    else:
        raise ValueError("Wrong method: {}".format(METHOD))
