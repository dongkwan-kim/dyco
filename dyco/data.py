from argparse import Namespace
from typing import Type, Any, Optional, Union, Dict, Tuple
from pprint import pprint

import torch
from termcolor import cprint

import torch_geometric.transforms as T
from torch.utils.data import DataLoader
from torch_geometric.data import InMemoryDataset, Data, TemporalData
from torch_geometric.datasets import JODIEDataset
from pytorch_lightning import LightningDataModule
from torch_geometric.transforms import ToUndirected, Compose, AddSelfLoops
from torch_geometric.utils import add_self_loops

from data_loader import SnapshotGraphLoader, EdgeLoader
from data_utils import ToTemporalData, UseValEdgesAsInput, ToSymSparseTensor, ToSymmetric, FromTemporalData
from dataset import get_dynamic_graph_dataset, SingletonICEWS18, SingletonGDELT, DatasetType
from utils import try_getattr, get_log_func, idx_to_mask, EternalIter


class DyGraphDataModule(LightningDataModule):

    @property
    def h(self):
        return self.hparams

    def __init__(self,
                 dataset_name: str,
                 dataset_path: str,
                 dataloader_type: str,
                 batch_size: int,
                 eval_dataloader_type=None,
                 eval_batch_size=None,
                 step_size=1,
                 use_temporal_data=False,
                 use_sparse_tensor=False,
                 verbose=2,
                 num_workers=0,
                 prepare_data=False,
                 log_func=None,
                 *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=["prepare_data", "logger"])
        self._dataset: Optional[DatasetType] = None
        self.train_data, self.val_data, self.test_data = None, None, None
        self.split_idx: Union[Dict, None] = None
        self.model_kwargs = dict()

        if prepare_data:
            self.prepare_data()
        self.setup()

        self.log_func = log_func or get_log_func(cprint, color="green")
        if self.h.verbose >= 1:
            self.log_func(f"{self.__class__.__name__}/{self.h.dataset_name}: prepared and set up!")

    @property
    def split_edge(self) -> dict:
        return self.split_idx

    @property
    def dataset(self):
        return self._dataset[0] if isinstance(self._dataset, tuple) else self._dataset

    @property
    def num_classes(self):
        if hasattr(self.dataset, "task_type") and self.dataset.task_type == "link prediction":
            return 2
        elif isinstance(self.dataset, (SingletonICEWS18, SingletonGDELT)):
            # The task for them is a multi-class classification against
            # the corresponding object and subject entities.
            # Ref. https://github.com/rusty1s/pytorch_geometric/blob/master/examples/renet.py#L50-L51
            return self.dataset.num_nodes
        else:
            return self.dataset.num_classes

    @property
    def num_nodes(self) -> int:
        return self.dataset.num_nodes

    @property
    def num_rels(self) -> int:
        return self.dataset.num_rels if hasattr(self.dataset, "num_rels") else 0

    @property
    def num_node_features(self):
        try:
            return self.dataset.num_node_features
        except AttributeError:
            return 0

    @property
    def dataset_kwargs(self) -> Dict:
        if self.h.dataset_name in ["SingletonICEWS18", "SingletonGDELT"]:
            if self.h.use_temporal_data:
                return {"splits": ["train", "val", "test"]}
            else:
                return {"splits": ["train", "train_val", "train_val_test"]}
        else:
            return {}

    def prepare_data(self) -> None:
        get_dynamic_graph_dataset(path=self.h.dataset_path, name=self.h.dataset_name,
                                  **self.dataset_kwargs)

    def setup(self, stage: Optional[str] = None) -> None:

        tfs = []
        if self.h.dataset_name == "ogbn-arxiv":
            tfs.append(AddSelfLoops())  # important.
            if not self.h.use_temporal_data:
                tfs.append(ToSymmetric(to_sparse_tensor=self.h.use_sparse_tensor))
            else:
                tfs.append(ToTemporalData())
        elif self.h.dataset_name.startswith("JODIEDataset"):
            if not self.h.use_temporal_data:
                tfs.append(FromTemporalData())
        elif self.h.use_temporal_data:
            tfs.append(ToTemporalData())
        elif self.h.dataset_name == "ogbl-collab":
            if not self.h.use_temporal_data:
                tfs.append(UseValEdgesAsInput(to_sparse_tensor=self.h.use_sparse_tensor, to_symmetric=True))

        transform = None if len(tfs) == 0 else Compose(tfs)
        self._dataset = get_dynamic_graph_dataset(
            path=self.h.dataset_path, name=self.h.dataset_name, transform=transform,
            **self.dataset_kwargs,
        )
        if self.h.dataset_name.startswith("JODIEDataset"):
            try:  # TemporalData
                self.train_data, self.val_data, self.test_data = self._dataset[0].train_val_test_split(
                    val_ratio=0.15, test_ratio=0.15)
            except AttributeError:  # Data
                # todo: support split for Data
                raise NotImplementedError
        elif self.h.dataset_name in ["SingletonICEWS18", "SingletonGDELT"]:
            self.train_data, self.val_data, self.test_data = (d[0] for d in self._dataset)
            if not self.h.use_temporal_data:
                self.split_idx = self._dataset[0].get_rel_split()
        elif self.h.dataset_name == "ogbn-arxiv":
            self.split_idx = self._dataset.get_idx_split()
            node_split_mask = idx_to_mask(self.split_idx, self._dataset.num_nodes)
            self.train_data, self.val_data, self.test_data = self._dataset[0], self._dataset[0], self._dataset[0]
            self.train_data.train_mask = node_split_mask["train"]
            self.val_data.val_mask = node_split_mask["valid"]
            self.test_data.test_mask = node_split_mask["test"]
            self.model_kwargs["add_self_loops"] = False  # important.
        elif self.h.dataset_name == "ogbl-collab":
            self.split_idx = self._dataset.get_edge_split()
            UseValEdgesAsInput.set_val_edge_index_for_compose(transform, self.split_idx)
            self.train_data, self.val_data, self.test_data = self._dataset[0], self._dataset[0], self._dataset[0]
        elif self.h.dataset_name == "BitcoinOTC":
            raise NotImplementedError
        else:
            raise ValueError

    def train_dataloader(self):
        if self.h.use_temporal_data and self.h.dataloader_type == "TemporalDataLoader":
            # Implement https://github.com/rusty1s/pytorch_geometric/blob/master/examples/tgn.py
            raise NotImplementedError
        elif self.h.dataloader_type == "SnapshotGraphLoader":
            loader = SnapshotGraphLoader(
                data=self.train_data,
                loading_type=SnapshotGraphLoader.get_loading_type(self.h.dataset_name),
                batch_size=self.h.batch_size,
                step_size=self.h.step_size,
                shuffle=True,
                num_workers=self.h.num_workers,
                transform_after_collation=T.ToSparseTensor(fill_cache=False) if self.h.use_sparse_tensor else None,
                edge_batch_size=self.h.get("edge_batch_size"),
                **SnapshotGraphLoader.get_kwargs_from_dataset(self.dataset),  # snapshot_dir, num_nodes
            )
        elif self.h.dataloader_type == "EdgeLoader":
            pos_train_edge = self.split_edge["train"]["edge"]
            kwargs = try_getattr(self.train_data, ["x", "adj_t", "edge_index"])
            loader = EdgeLoader(batch_size=self.h.batch_size,
                                pos_edge_index=pos_train_edge, neg_edge_index="trivial_random_samples",
                                num_nodes=self.num_nodes, additional_kwargs=kwargs, shuffle=True)
        elif self.h.dataloader_type == "NoLoader":
            loader = EternalIter([self.train_data])
        else:
            raise ValueError("Wrong options: ({}, use_temporal_data={})".format(
                self.h.dataloader_type, self.h.use_temporal_data))
        return loader

    def _eval_loader(self, eval_data: Union[Data, TemporalData], stage=None):
        dataloader_type = (self.h.eval_dataloader_type or self.h.dataloader_type)
        batch_size = (self.h.eval_batch_size or self.h.batch_size)
        if self.h.use_temporal_data and dataloader_type == "TemporalDataLoader":
            # Implement https://github.com/rusty1s/pytorch_geometric/blob/master/examples/tgn.py
            raise NotImplementedError
        elif dataloader_type == "SnapshotGraphLoader":
            loader = SnapshotGraphLoader(
                data=eval_data,
                loading_type=SnapshotGraphLoader.get_loading_type(self.h.dataset_name),
                batch_size=batch_size,
                step_size=self.h.step_size,
                shuffle=False,
                num_workers=self.h.num_workers,
                transform_after_collation=T.ToSparseTensor(fill_cache=False) if self.h.use_sparse_tensor else None,
                edge_batch_size=self.h.get("edge_batch_size"),
                **SnapshotGraphLoader.get_kwargs_from_dataset(self.dataset),  # snapshot_dir, num_nodes, ...
            )
        elif dataloader_type == "NoLoader":
            loader = EternalIter([eval_data])
        elif dataloader_type == "EdgeLoader":
            assert isinstance(eval_data, Data)  # todo: support TemporalData
            # Implement https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/collab/gnn.py#L140-L178
            pos_valid_edge = self.split_edge[stage]['edge']
            neg_valid_edge = self.split_edge[stage]['edge_neg']

            if stage == "valid":
                kwargs = try_getattr(eval_data, ["x", "adj_t", "edge_index"])
            else:  # test
                kwargs = try_getattr(eval_data, ["x", "full_adj_t", "full_edge_index"])

            loader = EdgeLoader(batch_size=batch_size,
                                pos_edge_index=pos_valid_edge, neg_edge_index=neg_valid_edge,
                                num_nodes=self.num_nodes,
                                additional_kwargs=kwargs, batch_idx_to_add_kwargs=0,
                                shuffle=False)
        else:
            raise ValueError("Wrong options: ({}, use_temporal_data={})".format(
                dataloader_type, self.h.use_temporal_data))
        return loader

    def val_dataloader(self):
        return self._eval_loader(self.val_data, stage="valid")

    def test_dataloader(self):
        return self._eval_loader(self.test_data, stage="test")

    def __repr__(self):
        return "{}(dataset={})".format(self.__class__.__name__, self.h.dataset_name)


if __name__ == '__main__':

    from pytorch_lightning import seed_everything
    seed_everything(32)
    # JODIEDataset/reddit, JODIEDataset/wikipedia, JODIEDataset/mooc, JODIEDataset/lastfm
    # ogbn-arxiv, ogbl-collab, ogbl-citation2
    # SingletonICEWS18, SingletonGDELT
    # BitcoinOTC
    NAME = "ogbl-collab"
    LOADER = "SnapshotGraphLoader"  # SnapshotGraphLoader, TemporalDataLoader, EdgeLoader, NoLoader
    EVAL_LOADER = "EdgeLoader"  # + None
    USE_TEMPORAL_DATA = False

    if LOADER == "EdgeLoader" or EVAL_LOADER == "EdgeLoader":
        assert NAME == "ogbl-collab"

    _dgdm = DyGraphDataModule(
        dataset_name=NAME,
        dataset_path="/mnt/nas2/GNN-DATA/PYG/",
        dataloader_type=LOADER,
        batch_size=8,
        eval_dataloader_type=EVAL_LOADER,
        eval_batch_size=None,
        step_size=1,
        use_temporal_data=USE_TEMPORAL_DATA,
        use_sparse_tensor=True,
        verbose=2,
        num_workers=0,
        edge_batch_size=69001,
    )
    print(_dgdm)
    print("model_kwargs", _dgdm.model_kwargs)
    cprint("Train ----", "green")
    for _i, _b in enumerate(_dgdm.train_dataloader()):
        pprint(_b)
        try:
            print("\t - edge_index", _b.edge_index.min(), _b.edge_index.max())
            print("\t - train_edge_index", _b.train_edge_index.min(), _b.train_edge_index.max())
        except AttributeError:
            pass
        if _i == 2:
            break
    cprint("Valid ----", "green")
    for _i, _b in enumerate(_dgdm.val_dataloader()):
        pprint(_b)
        if _i == 2:
            break
    cprint("Test ----", "green")
    for _i, _b in enumerate(_dgdm.test_dataloader()):
        pprint(_b)
        if _i == 2:
            break
