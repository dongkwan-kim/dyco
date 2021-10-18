from argparse import Namespace
from pprint import pprint
from typing import Dict, Union, Optional, Tuple, Any, List

import torch
import torch.nn as nn
from pytorch_lightning import (LightningModule)
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import Tensor
from torch_geometric.data import Batch

from data import DyGraphDataModule
from data_utils import BatchType
from evaluator import VersatileGraphEvaluator
from model_loss import BCEWOLabelsLoss, InfoNCEWithReadoutLoss
from model_utils import GraphEncoder, VersatileEmbedding, MLP, EdgePredictor, Readout
from utils import try_getattr, ld_to_dl, iter_ft
from run_utils import get_logger

log = get_logger(__name__)


def x_and_edge(batch) -> Dict[str, Tensor]:
    x = getattr(batch, "x", None)
    edge_index = try_getattr(batch, ["adj_t", "edge_index"], None,
                             iter_all=False, as_dict=False)[0]
    rel = getattr(batch, "rel", None)
    if rel is not None:
        # edge_index=[sub, obj] is used to predict obj_log_prob, sub_log_prob.
        pred_edges = {"obj": batch.edge_index[0], "sub": batch.edge_index[1]}
    else:
        pred_edges = try_getattr(batch, ["pos_edge", "neg_edge"], {},
                                 iter_all=True, as_dict=True)
        pred_edges.pop("default", None)
    return {"x": x, "edge_index": edge_index, "rel": rel, **pred_edges}


class StaticGraphModel(LightningModule):

    @property
    def h(self):
        return self.hparams

    @property
    def dh(self):
        return self.given_datamodule.hparams

    def __init__(self,
                 node_embedding_type: str,
                 num_node_emb_channels: int,
                 edge_embedding_type: str,
                 num_edge_emb_channels: int,
                 encoder_layer_name: str,
                 num_layers: int,
                 hidden_channels: int,
                 out_channels: Optional[int],
                 activation: str,
                 use_bn: bool,
                 use_skip: bool,
                 dropout_channels: float,
                 layer_kwargs: Dict[str, Any],
                 use_projector: bool,
                 projected_channels: int,
                 projector_infonce_temperature: float,
                 projector_readout_types: str,
                 use_predictor: bool,
                 predictor_type: str,
                 metrics: List[str],
                 lambda_pred: float,
                 lambda_proj: float,
                 learning_rate: float,
                 weight_decay: float,
                 subname: str = "default",
                 given_datamodule: DyGraphDataModule = None):
        super().__init__()
        self.save_hyperparameters(ignore=["given_datamodule"])
        assert given_datamodule is not None
        self.given_datamodule = given_datamodule

        if given_datamodule.num_node_features > 0:
            node_embedding_type, num_node_emb_channels = "UseRawFeature", given_datamodule.num_node_features
        else:
            node_embedding_type, num_node_emb_channels = self.h.node_embedding_type, self.h.num_node_emb_channels
        self.node_emb = VersatileEmbedding(
            embedding_type=node_embedding_type,
            num_entities=given_datamodule.num_nodes,
            num_channels=num_node_emb_channels,
        )
        use_edge_emb = given_datamodule.num_rels > 0
        if use_edge_emb:
            self.edge_emb = VersatileEmbedding(
                embedding_type=self.h.edge_embedding_type,
                num_entities=given_datamodule.num_rels,
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
            **given_datamodule.model_kwargs,
        )

        self.projector, self.predictor = None, None
        self.projector_loss, self.predictor_loss = None, None

        if self.h.use_projector:
            self.projector = MLP(
                num_layers=2,
                in_channels=self.h.hidden_channels,
                hidden_channels=self.h.hidden_channels,
                out_channels=self.h.projected_channels,
                activation=self.h.activation,
                use_bn=self.h.use_bn,
                dropout=self.h.dropout_channels,
                activate_last=False,
            )
            _use_out_linear = "-" in self.h.projector_readout_types  # multi-readout
            self.projector_loss = InfoNCEWithReadoutLoss(
                temperature=self.h.projector_infonce_temperature,
                readout=Readout(readout_types=self.h.projector_readout_types,
                                use_in_mlp=False, use_out_linear=_use_out_linear,
                                hidden_channels=self.h.projected_channels,
                                out_channels=self.h.projected_channels))

        # e.g., Node, Edge/DotProduct, Edge/Concat, Edge/HadamardProduct
        if self.h.predictor_type.lower().startswith("node"):
            self.predictor = MLP(
                num_layers=2,
                in_channels=self.h.hidden_channels,
                hidden_channels=self.h.hidden_channels,
                out_channels=given_datamodule.num_classes,
                activation=self.h.activation,
                use_bn=self.h.use_bn,
                dropout=self.h.dropout_channels,
                activate_last=False,
            )
            self.predictor_loss = nn.CrossEntropyLoss()
        elif self.h.predictor_type.lower().startswith("edge"):
            _, predictor_type = self.h.predictor_type.split("/")
            out_channels = 1 if given_datamodule.num_classes == 2 else given_datamodule.num_classes
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
            if out_activation == "sigmoid":
                self.predictor_loss = BCEWOLabelsLoss()
            else:
                self.predictor_loss = nn.CrossEntropyLoss()

        self._encoded_x: Optional[Tensor] = None

        self.evaluator = VersatileGraphEvaluator(
            name=self.dh.dataset_name,
            metrics=self.h.metrics,
        )

    def forward(self,
                x, edge_index, rel=None,
                encoded_x=None,
                use_predictor=True,
                use_projector=True,
                return_encoded_x=False,
                pred_edges: Dict[str, Tensor] = None
                ) -> Dict[str, Tensor]:

        encoder_kwargs = {}
        if encoded_x is None:
            x = self.node_emb(x)
            if rel is not None:
                encoder_kwargs["edge_attr"] = self.edge_emb(rel)
            x = self.encoder(x, edge_index, **encoder_kwargs)
        else:
            x, edge_attr = encoded_x, None

        out = dict()
        if return_encoded_x:
            out["encoded_x"] = x
        if use_projector:
            assert self.projector is not None  # MLP
            out["proj_x"] = self.projector(x)
        if use_predictor:  # MLP, EdgePredictor
            assert self.predictor is not None
            if pred_edges is None:
                out["log_prob"] = self.predictor(x, edge_index)
            else:
                # {pos_edge, neg_edge} or {obj, sub}
                for pred_name, _pred_edge in pred_edges.items():
                    if rel is not None:
                        _prob = self.predictor(x=x, edge_index=_pred_edge)
                    else:
                        # [ x[sub/obj] || rel ] --> obj/sub_log_prob
                        _prob = self.predictor(x_i=x[_pred_edge], x_j=edge_attr)
                    out[f"{pred_name}_prob"] = _prob

        # Dict of encoded_x, proj_x, log_prob, pos_edge_prob, neg_edge_prob
        return out

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(),
                                lr=self.h.learning_rate, weight_decay=self.h.weight_decay)

    def get_pred_loss_and_results(
            self,
            batch: BatchType,
            out: Dict[str, Tensor],
            x_and_edge_kwargs: Dict[str, Tensor],
    ) -> Tuple[Optional[Tensor], Optional[Dict[str, Tensor]]]:

        # ogbn-arxiv: CrossEntropyLoss(logits[mask], y[mask])
        if "log_prob" in out:
            mask = try_getattr(batch, ["train_mask", "val_mask", "test_mask"], iter_all=False, as_dict=False)[0]
            log_prob = out["log_prob"]
            y = batch.y.squeeze(1)
            rets = {"y_pred": log_prob[mask], "y_true": y[mask]}
            loss = self.predictor_loss(rets["y_pred"], rets["y_true"])

        # ogbl-collab: BCEWOLabelsLoss(pos_edge_prob, neg_edge_prob)
        elif "pos_edge_prob" in out:
            pos_edge_prob, neg_edge_prob = out["pos_edge_prob"], out["neg_edge_prob"]
            rets = {"y_pred_pos": pos_edge_prob, "y_pred_neg": neg_edge_prob}
            loss = self.predictor_loss(pos_edge_prob, neg_edge_prob)

        # Temporal-KG: CrossEntropyLoss(obj_log_prob, obj_node) + CrossEntropyLoss(sub_log_prob, sub_node)
        elif "obj_log_prob" in out:
            mask = try_getattr(batch, ["val_mask", "test_mask"], default=None, iter_all=False, as_dict=False)[0]
            # edge_index=[sub, obj] is used to predict obj_log_prob, sub_log_prob.
            sub_node, obj_node = x_and_edge_kwargs["obj"], x_and_edge_kwargs["sub"]
            obj_log_prob, sub_log_prob = out["obj_log_prob"], out["sub_log_prob"]

            rets = {"obj_pred": obj_log_prob, "obj_node": obj_node,
                    "sub_pred": sub_log_prob, "sub_node": sub_node}
            if mask is not None:
                for r_key, r_tensor in rets.items():
                    rets[r_key] = r_tensor[mask]

            obj_pred_loss = self.predictor_loss(rets["obj_pred"], rets["obj_node"])
            sub_pred_loss = self.predictor_loss(rets["sub_pred"], rets["sub_node"])
            loss = obj_pred_loss + sub_pred_loss

        else:
            loss, rets = None, None

        return loss, rets

    def get_proj_loss(self, batch: BatchType, out: Dict[str, Tensor]) -> Optional[Tensor]:
        if "proj_x" in out:
            return self.projector_loss(out["proj_x"], batch.batch)
        else:
            return None

    def step(self, batch: BatchType, batch_idx: int,
             use_predictor: bool, use_projector: bool, cache_encoded_x: bool):
        """
        :param batch: Keys are one of
            batch=[N], edge_index=[2, E], adj_t=[N, N, nnz],
            neg_edge=[2, B], pos_edge=[2, B], rel=[E, 1], train_mask=[N],
            x=[N, F], y=[N, 1] and ptr, t.
        :param batch_idx:
        :param use_predictor:
        :param use_projector:
        :param cache_encoded_x:
        :return:
        """
        x_and_edge_kwargs = x_and_edge(batch)
        out = self.forward(
            **x_and_edge_kwargs,  # x, edge_index, (and rel, pred_edges)
            use_predictor=use_predictor,
            use_projector=use_projector,
            return_encoded_x=cache_encoded_x,
        )
        pred_loss, pred_rets = self.get_pred_loss_and_results(batch, out, x_and_edge_kwargs)
        proj_loss = self.get_proj_loss(batch, out)

        if cache_encoded_x and batch_idx == 0:
            assert self._encoded_x is None
            assert "encoded_x" in out
            self._encoded_x = out["encoded_x"]

        loss_parts, total_loss = {}, 0
        if pred_loss is not None:
            total_loss += self.h.lambda_pred * pred_loss
            loss_parts["pred_loss"] = pred_loss
        if proj_loss is not None:
            total_loss += self.h.lambda_proj * proj_loss
            loss_parts["proj_loss"] = proj_loss
        return {"loss": total_loss, **loss_parts, **pred_rets}

    def training_step(self, batch: BatchType, batch_idx: int):
        return self.step(
            batch=batch, batch_idx=batch_idx,
            use_predictor=self.h.use_predictor,
            use_projector=self.h.use_projector,
            cache_encoded_x=False,
        )

    def validation_step(self, batch: BatchType, batch_idx: int):
        eval_dataloader_type = self.dh.eval_dataloader_type or self.dh.dataloader_type
        return self.step(
            batch=batch, batch_idx=batch_idx,
            use_predictor=self.h.use_predictor,
            use_projector=self.h.use_projector and (eval_dataloader_type == "SnapshotGraphLoader"),
            cache_encoded_x=(eval_dataloader_type == "EdgeLoader"),
        )

    def test_step(self, batch: BatchType, batch_idx: int):
        eval_dataloader_type = self.dh.eval_dataloader_type or self.dh.dataloader_type
        return self.step(
            batch=batch, batch_idx=batch_idx,
            use_predictor=True,
            use_projector=False,
            cache_encoded_x=(eval_dataloader_type == "EdgeLoader"),
        )

    def epoch_end(self, prefix, outputs, idx_of_hp_metric=None):
        self._encoded_x = None  # cache flush

        output_as_dict = ld_to_dl(outputs)

        for loss_name, loss_val in iter_ft(output_as_dict.items(),
                                           transform=lambda kv: (kv[0], torch.stack(kv[1]).mean()),
                                           condition=lambda kv: ("loss" in kv[0])):
            self.log(f"{prefix}/{loss_name}", loss_val, prog_bar=False)

        eval_rets = self.evaluator.eval(input_dict=dict(iter_ft(
            output_as_dict.items(),
            transform=lambda kv: (kv[0], torch.cat(kv[1])),
            condition=lambda kv: ("loss" not in kv[0]))))
        for i, (metric, value) in enumerate(eval_rets.items()):
            self.log(f"{prefix}/{metric}", value, prog_bar=True)
            if idx_of_hp_metric is not None and i == idx_of_hp_metric:
                self.logger.log_metrics({"hp_metric": float(value)})
                log.info(f"Log '{metric}' as hp_metric")

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.epoch_end("train", outputs)

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.epoch_end("valid", outputs)

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.epoch_end("test", outputs, idx_of_hp_metric=0)


if __name__ == '__main__':
    NAME = "ogbl-collab"
    USE_TEMPORAL_DATA = False
    LOADER = "SnapshotGraphLoader"  # SnapshotGraphLoader, TemporalDataLoader, EdgeLoader, NoLoader
    EVAL_LOADER = "NoLoader"  # + None
    # JODIEDataset/reddit, JODIEDataset/wikipedia, JODIEDataset/mooc, JODIEDataset/lastfm
    # ogbn-arxiv, ogbl-collab, ogbl-citation2
    # SingletonICEWS18, SingletonGDELT
    # BitcoinOTC
    if LOADER == "EdgeLoader" or EVAL_LOADER == "EdgeLoader":
        assert NAME == "ogbl-collab"

    _dgdm = DyGraphDataModule(
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
    )
    print(_dgdm)
    print("model_kwargs", _dgdm.model_kwargs)

    _sgm = StaticGraphModel(
        node_embedding_type="Embedding",
        num_node_emb_channels=128,
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

        use_projector=True,
        projected_channels=128,
        projector_infonce_temperature=0.5,
        projector_readout_types="mean-max",

        use_predictor=True,
        predictor_type="Edge/HadamardProduct",
        metrics=["hits@10", "hits@50", "hits@100"],

        lambda_pred=1.0,
        lambda_proj=1.0,

        learning_rate=1e-3,
        weight_decay=1e-5,
        given_datamodule=_dgdm,
    )
    print(_sgm)
