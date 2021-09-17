import time
from functools import reduce
from pprint import pprint
from typing import Dict, Any, List, Tuple, Optional, Callable, Union

import torch
from termcolor import cprint
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.utils import subgraph, to_undirected
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import to_dense_batch, softmax

from torch_scatter import scatter

import numpy as np
from tqdm import tqdm

__MAGIC__ = "This is magic, please trust the author."


def try_getattr(o, name_list: List[str],
                default=__MAGIC__, iter_all=True,
                as_dict=True) -> Union[Dict[str, Any], List, Any]:
    ret_list = list()
    for name in name_list:
        try:
            ret_list.append(getattr(o, name))
            if not iter_all:
                break
        except AttributeError:
            pass
    if len(ret_list) > 0:
        if as_dict:
            return dict(zip(name_list, ret_list))
        else:
            return ret_list
    elif default != __MAGIC__:
        if as_dict:
            return {"default": default}
        else:
            return default
    else:
        raise AttributeError


def iter_transform(iterator, transform: Callable = None):
    for it in iterator:
        yield it if transform is None else transform(it)


def merge_dict_by_keys(first_dict: dict, second_dict: dict, keys: list):
    for k in keys:
        if k in second_dict:
            first_dict[k] = second_dict[k]
    return first_dict


def merge_dict(dict_list: List[Dict]) -> Dict:
    # https://stackoverflow.com/a/16048368
    return reduce(lambda a, b: dict(a, **b), dict_list)


def merge_dict_by_reducing_values(dict_list: List[Dict], reduce_values: Callable = sum):
    def _reduce(the_dict, a_dict):
        for k, v in a_dict.items():
            if k in the_dict:
                v = reduce_values([v, the_dict[k]])
            the_dict[k] = v
        return the_dict

    return reduce(_reduce, dict_list, {})


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


def rename_attr(obj, old_name, new_name):
    # https://stackoverflow.com/a/25310860
    obj.__dict__[new_name] = obj.__dict__.pop(old_name)


def func_compose(*funcs):
    # compose(f1, f2, f3)(x) == f3(f2(f1(x)))
    # https://stackoverflow.com/a/16739663
    funcs = [_f for _f in funcs if _f is not None]
    return lambda x: reduce(lambda acc, f: f(acc), funcs, x)


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


def add_self_loops_v2(edge_index, edge_weight: Optional[torch.Tensor] = None,
                      edge_attr: Optional[torch.Tensor] = None, edge_attr_reduce: str = "mean",
                      fill_value: float = 1., num_nodes: Optional[int] = None):
    r"""Extended method of torch_geometric.utils.add_self_loops that
    supports :attr:`edge_attr`."""
    N = maybe_num_nodes(edge_index, num_nodes)

    loop_index = torch.arange(0, N, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        loop_weight = edge_weight.new_full((N,), fill_value)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

    if edge_attr is not None:
        assert edge_attr.size(0) == edge_index.size(1)
        if edge_attr_reduce != "fill":
            loop_attr = scatter(edge_attr, edge_index[0], dim=0, dim_size=N,
                                reduce=edge_attr_reduce)
        else:
            loop_attr = edge_attr.new_full((N, edge_attr.size(1)), fill_value)
        edge_attr = torch.cat([edge_attr, loop_attr], dim=0)

    edge_index = torch.cat([edge_index, loop_index], dim=1)

    return edge_index, edge_weight, edge_attr


def idx_to_mask(idx_dict: Dict[Any, Tensor], num_nodes: int):
    mask_dict = dict()
    for k, idx in idx_dict.items():
        # idx: LongTensor
        mask = torch.zeros((num_nodes,), dtype=torch.bool)
        mask[idx] = 1
        mask_dict[k] = mask
    return mask_dict


if __name__ == '__main__':

    METHOD = "merge_dict_by_reducing_values"

    from pytorch_lightning import seed_everything

    seed_everything(42)

    if METHOD == "to_index_chunks_by_values":
        _tensor_1d = torch.Tensor([24, 20, 21, 21, 20, 23, 24])
        print(to_index_chunks_by_values(_tensor_1d))

    if METHOD == "merge_dict_by_reducing_values":
        pprint(merge_dict_by_reducing_values(
            [{"a": 1, "c": 3},
             {"a": 10, "b": 20, "c": 30},
             {"a": 100, "b": 200}],
            reduce_values=sum))

    elif METHOD == "add_self_loops_v2":
        _edge_index = torch.Tensor([[0, 0, 1],
                                    [1, 2, 2]]).long()
        _edge_attr = torch.eye(3).float()

        _edge_index, _edge_attr = to_undirected(_edge_index, _edge_attr)
        print(_edge_index)
        print(_edge_attr)
        print("-" * 7)

        _edge_index, _, _edge_attr = add_self_loops_v2(
            edge_index=_edge_index, edge_attr=_edge_attr, edge_attr_reduce="sum")
        print(_edge_index)
        print(_edge_attr)

    else:
        raise ValueError("Wrong method: {}".format(METHOD))
