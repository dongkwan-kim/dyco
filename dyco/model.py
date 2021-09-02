from argparse import Namespace

import torch
import torch.nn as nn
from pytorch_lightning import (LightningModule)
from torch import Tensor
from torch_geometric.data import Batch

from data import DyGraphDataModule
from model_utils import GraphEncoder, VersatileEmbedding, MLP, EdgePredictor


class StaticGraphModel(LightningModule):

    @property
    def h(self):
        return self.hparams

    def __init__(self, hparams, data_module: DyGraphDataModule):  # todo
        super().__init__()
        self.save_hyperparameters(hparams)

        if data_module.num_node_features > 0:
            node_embedding_type, num_node_emb_channels = "UseRawFeature", data_module.num_node_features
        else:
            node_embedding_type, num_node_emb_channels = self.h.node_embedding_type, self.h.num_node_emb_channels
        self.node_emb = VersatileEmbedding(
            embedding_type=node_embedding_type,
            num_entities=data_module.num_nodes,
            num_channels=num_node_emb_channels,
        )
        if self.h.use_edge_emb:
            self.edge_emb = VersatileEmbedding(
                embedding_type=self.h.edge_embedding_type,
                num_entities=data_module.num_rels,
                num_channels=self.h.num_edge_emb_channels,
            )
        else:
            self.edge_emb = None

        self.encoder = GraphEncoder(
            layer_name=self.h.encoder_layer_name,
            num_layers=self.h.num_layers,
            in_channels=num_node_emb_channels,
            hidden_channels=self.h.hidden_channels,
            out_channels=(self.h.out_channels or self.h.hidden_channels),  # use hidden if not given.
            activation=self.h.activation,
            use_bn=self.h.use_bn,
            use_skip=self.h.use_skip,
            dropout_channels=self.h.dropout_channels,
            **self.h.layer_kwargs,
            **data_module.model_kwargs,
        )

        self.projector, self.predictor = None, None

        if self.h.use_projection:
            self.projector = MLP(
                num_layers=2,
                in_channels=self.h.hidden_channels,
                hidden_channels=self.h.hidden_channels,
                out_channels=self.h.hidden_channels,
                activation=self.h.activation,
                use_bn=self.h.use_bn,
                dropout=self.h.dropout_channels,
                activate_last=False,
            )

        # e.g., Node, Edge/DotProduct, Edge/Concat, Edge/HadamardProduct
        if self.h.predictor_type.lower().startswith("node"):
            self.predictor = MLP(
                num_layers=2,
                in_channels=self.h.hidden_channels,
                hidden_channels=self.h.hidden_channels,
                out_channels=data_module.num_classes,
                activation=self.h.activation,
                use_bn=self.h.use_bn,
                dropout=self.h.dropout_channels,
                activate_last=False,
            )
        elif self.h.predictor_type.lower().startswith("edge"):
            _, predictor_type = self.h.predictor_type.split("/")
            out_channels = 1 if data_module.num_classes == 2 else data_module.num_classes
            out_activation = "sigmoid" if out_channels == 1 else None
            self.predictor = EdgePredictor(
                predictor_type=predictor_type,
                num_layers=2,
                hidden_channels=self.h.hidden_channels,
                out_channels=out_channels,
                activation=self.h.activation,
                out_activation=out_activation,
                use_bn=self.h.use_bn,
                dropout_channels=self.h.dropout_channels,
            )

    def forward(self, x, edge_index) -> Tensor:
        x = self.node_emb(x)
        x = self.encoder(x, edge_index)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(),
                                lr=self.h.learning_rate, weight_decay=self.h.weight_decay)


if __name__ == '__main__':
    NAME = "SingletonICEWS18"
    USE_TEMPORAL_DATA = False
    LOADER = "SnapshotGraphLoader"  # SnapshotGraphLoader, TemporalDataLoader, EdgeLoader, NoLoader
    EVAL_LOADER = "NoLoader"  # + None
    # JODIEDataset/reddit, JODIEDataset/wikipedia, JODIEDataset/mooc, JODIEDataset/lastfm
    # ogbn-arxiv, ogbl-collab, ogbl-citation2
    # SingletonICEWS18, SingletonGDELT
    # BitcoinOTC
    if LOADER == "EdgeLoader" or EVAL_LOADER == "EdgeLoader":
        assert NAME == "ogbl-collab"

    _dgdm = DyGraphDataModule(Namespace(
        verbose=2,
        dataset_name=NAME,
        dataset_path="/mnt/nas2/GNN-DATA/PYG/",
        dataloader_type=LOADER,
        eval_dataloader_type=EVAL_LOADER,
        use_temporal_data=USE_TEMPORAL_DATA,
        use_sparse_tensor=True,
        batch_size=8,
        eval_batch_size=None,
        step_size=1,
        num_workers=0,
    ))
    print(_dgdm)
    print("model_kwargs", _dgdm.model_kwargs)

    _sgm = StaticGraphModel(
        hparams=Namespace(
            node_embedding_type="Embedding",
            num_node_emb_channels=128,
            use_edge_emb=True,
            edge_embedding_type="Embedding",
            num_edge_emb_channels=128,
            encoder_layer_name="GATConv",
            num_layers=3,
            hidden_channels=256,
            out_channels=None,
            activation="relu",
            use_bn=False,
            use_skip=True,
            dropout_channels=0.5,
            layer_kwargs={"heads": 8},
            use_projection=True,
            predictor_type="Edge/HadamardProduct",
        ),
        data_module=_dgdm,
    )
    print(_sgm)
