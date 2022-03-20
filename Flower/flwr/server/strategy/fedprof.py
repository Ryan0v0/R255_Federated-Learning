# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Federated Averaging (FedAvg) [McMahan et al., 2016] strategy.

Paper: https://arxiv.org/abs/1602.05629
"""


from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple

import copy
import glob
import logging
import math
import os
import random
import sys

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn

import wandb

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from .aggregate import aggregate, weighted_loss_avg
from .strategy import Strategy
from .fedavg import FedAvg
from .footprinter import FootPrinter


DEPRECATION_WARNING = """
DEPRECATION WARNING: deprecated `eval_fn` return format

    loss, accuracy

move to

    loss, {"accuracy": accuracy}

instead. Note that compatibility with the deprecated return format will be
removed in a future release.
"""

DEPRECATION_WARNING_INITIAL_PARAMETERS = """
DEPRECATION WARNING: deprecated initial parameter type

    flwr.common.Weights (i.e., List[np.ndarray])

will be removed in a future update, move to

    flwr.common.Parameters

instead. Use

    parameters = flwr.common.weights_to_parameters(weights)

to easily transform `Weights` to `Parameters`.
"""

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_eval_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_eval_clients`.
"""


class FedProf(FedAvg):
    """Configurable FedProf strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn: Optional[
            Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
    ) -> None:
        """Federated Averaging strategy.

        Implementation based on https://arxiv.org/abs/1602.05629

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. Defaults to 0.1.
        fraction_eval : float, optional
            Fraction of clients used during validation. Defaults to 0.1.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_eval_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        eval_fn : Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        """
        super().__init__()
        # self.pred_credibility = np.array([0.0] * self.args.client_num_in_total)
        self.alpha = args.alpha
        self.beta = args.beta
        self.accept_failures = accept_failures
    
    def __repr__(self) -> str:
        rep = f"FedProf(accept_failures={self.accept_failures})"
        return rep

    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        
        footprinter = FootPrinter()
        
        # calculate credbility of each client
        footprinter.update_encoder(self.model_trainer.model.fc1)
        server_footprint = footprinter.culc_footprint(self.X_server, dataloader=False)
        
        for idx in range(self.args.client_num_in_total):
            client_footprint = footprinter.culc_footprint(self.train_data_local_dict[idx])
                      
            self.pred_credibility[idx] = math.e ** (-self.alpha * footprinter.kldiv_between_server_and_client(server_footprint, client_footprint))
        
        # sim_footprint = spearmanr(self.pred_credibility, self.true_credibility)[0]

        # Sample clients
        clients = self._fedprof_sampling(sample_size=sample_size, client_manager=client_manager)
                      
        # Prepare parameters and config
        config = {}
        if self.on_fit_config_fn is not None:
            # Use custom fit config function if provided
            config = self.on_fit_config_fn(rnd)
                      
        # Fit instructions
        fit_ins = FitIns(parameters, config)
                      
        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def _fedprof_sampling(
        self, sample_size: int, client_manager: ClientManager
    ) -> List[ClientProxy]:
        """Sample clients depending on their score from representation."""
        all_clients: Dict[str, ClientProxy] = client_manager.all()
        cid_idx: Dict[int, str] = {}
        raw: List[float] = []
        cpu_time: List[float] = []
        for idx, (cid, _) in enumerate(all_clients.items()):
            cid_idx[idx] = cid
            cpu_time[idx] = all_clients[idx].get_properties["cpu_time"]
        cpu_range = np.max(cpu_time) - np.min(cpu_time)

        pred_credibility = np.array([0.0] * len(all_clients.keys()))

        for idx, (cid, _) in enumerate(all_clients.items()):
            client = all_clients[idx]
            pred_credibility[cid] = math.e ** (-self.alpha* client.get_properties["kl_div"] - self.beta*((client.get_properties["cpu_time"]-np.min(cpu_time))/cpu_range)
                                                           
        # Sample clients
        return normalize_and_sample(
            all_clients=all_clients,
            cid_idx=cid_idx,
            sample_size=sample_size,
            pred_credibility=np.array(pred_credibility)
        )

def normalize_and_sample(
    all_clients: Dict[str, ClientProxy],
    cid_idx: Dict[int, str],
    sample_size: int,
    pred_credibility: np.ndarray,
) -> List[ClientProxy]:
    """Normalize the relative importance and sample clients accordingly.
        
    :meta private:
    """
    indices = np.arange(len(all_clients.keys()))
    sampled_indices = np.random.choice(
        indices, size=sample_size, replace=False, p=pred_credibility / np.sum(pred_credibility)
    )
    clients = [all_clients[cid_idx[idx]] for idx in sampled_indices]
    return clients

