from typing import Callable, Dict, Optional, Tuple

#from py.flwr.client.numpy_client import NumPyClient
from .fedavg import FedAvg
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple

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
from flwr.common.typing import PropertiesIns
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from .aggregate import aggregate, weighted_loss_avg
from .strategy import Strategy
import random
import numpy as np
import os

class FedSelect(FedAvg):
    """Configurable FedAdagrad strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes,too-many-locals
    def __init__(
        self,
        *,
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
        
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_eval=fraction_eval,
            min_fit_clients=min_fit_clients,
            min_eval_clients=min_eval_clients,
            min_available_clients=min_available_clients,
            eval_fn=eval_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
        )
        #self.current_weights = parameters_to_weights(initial_parameters)
        

    def __repr__(self) -> str:
        rep = f"FedOpt(accept_failures={self.accept_failures})"
        return rep
    
    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training based on the dataset size."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(rnd)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        
        # sample function
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = sample_size
        client_manager.wait_for(min_num_clients)
        # Sample clients which meet the criterion
        available_cids = list(client_manager.clients)

        # Sample clients which meet the criterion
        # Since getting prob each round is too time-consuming, so save the prob for one time and then load the prob after
        path_save = '/nfs-share/xinchi/XL-FL/amazon_prob.npy'
        if os.path.exists(path_save):
            print("LOADING PROBABILITY NUMPY ARRAY")
            prob = np.load(path_save)
        else:
            ins = PropertiesIns(config={})
            ins.config = {}
            num_samples_vect = [client_manager.clients[cid].get_properties(ins=ins).properties['num_samples'] for cid in available_cids]
            prob = np.array(num_samples_vect) / np.sum(num_samples_vect)
            np.save(path_save,prob)
        
        # sample the client
        sampled_cids = np.random.choice(available_cids, size=sample_size, replace=False, p=prob)
        clients = [client_manager.clients[cid] for cid in sampled_cids]


        clients = [client_manager.clients[cid] for cid in sampled_cids]
        # Return client/config pairs
        return [(client, fit_ins) for client in clients]
