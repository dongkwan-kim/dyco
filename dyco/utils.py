import time
from pprint import pprint
from typing import Dict, Any, List, Tuple, Optional

import torch
from termcolor import cprint
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.utils import subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import to_dense_batch, softmax

import numpy as np
from tqdm import tqdm


def merge_dict_by_keys(first_dict: dict, second_dict: dict, keys: list):
    for k in keys:
        if k in second_dict:
            first_dict[k] = second_dict[k]
    return first_dict


def startswith_any(string: str, prefix_list, *args, **kwargs) -> bool:
    for prefix in prefix_list:
        if string.startswith(prefix, *args, **kwargs):
            return True
    else:
        return False


def exist_attr(obj, name):
    return hasattr(obj, name) and (getattr(obj, name) is not None)


def del_attrs(o, keys: List[str]):
    for k in keys:
        delattr(o, k)


def debug_with_exit(func):  # Decorator
    def wrapped(*args, **kwargs):
        print()
        cprint("===== DEBUG ON {}=====".format(func.__name__), "red", "on_yellow")
        func(*args, **kwargs)
        cprint("=====   END  =====", "red", "on_yellow")
        exit()

    return wrapped


def print_time(method):  # Decorator
    """From https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d"""

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        cprint('%r  %2.2f s' % (method.__name__, (te - ts)), "red")
        return result

    return timed


# PyTorch/PyTorch Geometric related methods


def act(tensor, activation_name, **kwargs):
    if activation_name == "relu":
        return F.relu(tensor, **kwargs)
    elif activation_name == "elu":
        return F.elu(tensor, **kwargs)
    elif activation_name == "leaky_relu":
        return F.leaky_relu(tensor, **kwargs)
    elif activation_name == "sigmoid":
        return torch.sigmoid(tensor)
    elif activation_name == "tanh":
        return torch.tanh(tensor)
    else:
        raise ValueError(f"Wrong activation name: {activation_name}")


def get_extra_repr(model, important_args):
    return "\n".join(["{}={},".format(a, getattr(model, a)) for a in important_args
                      if a in model.__dict__])


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def torch_setdiff1d(tensor_1: Tensor, tensor_2: Tensor):
    dtype, device = tensor_1.dtype, tensor_1.device
    o = np.setdiff1d(tensor_1.numpy(), tensor_2.numpy())
    return torch.tensor(o, dtype=dtype, device=device)


def to_index_chunks_by_values(tensor_1d: Tensor, verbose=True) -> Dict[Any, Tensor]:
    tensor_1d = tensor_1d.flatten()
    index_chunks_dict = dict()
    # todo: there can be more efficient way, but it might not be necessary.
    v_generator = tqdm(torch.unique(tensor_1d), desc="index_chunks") if verbose else torch.unique(tensor_1d)
    for v in v_generator:
        v = v.item()
        index_chunk = torch.nonzero(tensor_1d == v).flatten()
        index_chunks_dict[v] = index_chunk
    return index_chunks_dict


def softmax_half(src: Tensor, index: Tensor, num_nodes: Optional[int] = None) -> Tensor:
    r"""softmax that supports torch.half tensors.
        See torch_geometric.utils.softmax for more details."""
    is_half = (src.dtype == torch.half)
    src = src.float() if is_half else src
    smx = softmax(src, index, num_nodes=num_nodes)
    return smx.half() if is_half else smx


def to_multiple_dense_batches(
        x_list: List[Tensor],
        batch=None, fill_value=0, max_num_nodes=None
) -> Tuple[List[Tensor], Tensor]:
    cat_x = torch.cat(x_list, dim=-1)
    cat_out, mask = to_dense_batch(cat_x, batch, fill_value, max_num_nodes)
    # [B, N, L*F] -> [B, N, F] * L
    return torch.chunk(cat_out, len(x_list), dim=-1), mask


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


def idx_to_mask(idx_dict: Dict[Any, Tensor], num_nodes: int):
    mask_dict = dict()
    for k, idx in idx_dict.items():
        # idx: LongTensor
        mask = torch.zeros((num_nodes,), dtype=torch.bool)
        mask[idx] = 1
        mask_dict[k] = mask
    return mask_dict


if __name__ == '__main__':

    METHOD = "sort_and_relabel"

    from pytorch_lightning import seed_everything
    seed_everything(42)

    if METHOD == "to_index_chunks_by_values":
        _tensor_1d = torch.Tensor([24, 20, 21, 21, 20, 23, 24])
        print(to_index_chunks_by_values(_tensor_1d))
    else:
        raise ValueError("Wrong method: {}".format(METHOD))
