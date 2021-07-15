import os
from typing import List

import torch
from termcolor import cprint
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from dyco.data import get_dynamic_dataset
from dyco.utils import to_index_chunks_by_values, subgraph_and_edge_mask


class DynamicGraphLoader(DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=True, num_workers=0,
                 **kwargs):

        self.dataset = dataset

        if len(dataset) == 1:  # ogbn, ogbl
            data = dataset[0]
            assert hasattr(data, "node_year") or hasattr(data, "edge_year"), "No node_year or edge_year"
            time_name = "node_year" if hasattr(data, "node_year") else "edge_year"
            self.data_list = self.disassemble_to_multi_snapshots(dataset[0], time_name, save_cache=True)
            # A complete snapshot at t is a cumulation of data_list from 0 -- t-1.
            self.require_cumulative_loading = True
        else:  # others:
            self.require_cumulative_loading = False
            raise NotImplementedError

        super(DynamicGraphLoader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
            collate_fn=self.__collate__, **kwargs,
        )

    @property
    def snapshot_path(self):
        return os.path.join(self.dataset.root, "snapshots.pt")

    def disassemble_to_multi_snapshots(self, data: Data, time_name: str,
                                       save_cache: bool = True) -> List[Data]:
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
                x_index = indices
            # index chunks are edges.
            # e.g., Data(edge_index=[2, E], edge_weight=[E, 1], edge_year=[E, 1], x=[E, F])
            elif "edge" in time_name:
                sub_edge_index = remained_edge_index[:, indices]
                x_index = torch.unique(sub_edge_index, sorted=False)
            else:
                raise ValueError(f"Wrong time_name: {time_name}")

            time_stamp = torch.Tensor([curr_time])
            data_at_curr = Data(time_stamp=time_stamp, edge_index=sub_edge_index, x_index=x_index)
            data_list.append(data_at_curr)

        if save_cache:
            cprint(f"Save snapshots at {self.snapshot_path}", "green")
            torch.save(data_list, self.snapshot_path)

        return data_list

    def __collate__(self, data_list):
        raise NotImplementedError


def get_dynamic_dataloader(dataset, name: str, stage: str, *args, **kwargs):
    assert stage in ["train", "valid", "test", "evaluation"]
    loader = DynamicGraphLoader(dataset, *args, **kwargs)
    return loader


if __name__ == '__main__':
    PATH = "/mnt/nas2/GNN-DATA/PYG/"
    # JODIEDataset/reddit, JODIEDataset/wikipedia, JODIEDataset/mooc, JODIEDataset/lastfm
    # ogbn-arxiv, ogbl-collab, ogbl-citation2
    # ICEWS18, GDELT
    # BitcoinOTC
    NAME = "ogbl-citation2"
    _d = get_dynamic_dataset(PATH, NAME)
    _loader = get_dynamic_dataloader(_d, NAME, "train")
