import logging
import os
import sys

from .utils import transform_grad_to_list

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML/")))
from fedml_api.distributed.fedavg.FedAvgServerManager import FedAVGServerManager
from fedml_api.distributed.fedavg.message_define import MyMessage
from fedml_api.distributed.fedavg.utils import (
    post_complete_message_to_sweep_process,
    transform_tensor_to_list,
)
from fedml_core.distributed.communication.message import Message
from fedml_core.distributed.server.server_manager import ServerManager


class SecureFedAVGServerManager(ServerManager):
    def __init__(
        self,
        args,
        aggregator,
        comm=None,
        rank=0,
        size=0,
        backend="MPI",
        is_preprocessed=False,
        preprocessed_client_lists=None,
    ):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.round_idx = 0
        self.is_preprocessed = is_preprocessed
        self.preprocessed_client_lists = preprocessed_client_lists
        self.client_indexes = []

    def run(self):
        super().run()

    def send_init_msg(self):
        self.sender_id_to_client_index = {}
        # sampling clients
        self.client_indexes = self.aggregator.client_sampling(
            self.round_idx,
            self.args.client_num_in_total,
            self.args.client_num_per_round,
        )
        global_model_params = self.aggregator.get_global_model_params()
        if self.args.is_mobile == 1:
            global_model_params = transform_tensor_to_list(global_model_params)

        for process_id in range(1, self.size):
            logging.info(
                f"send init message to client_index = {process_id - 1} (process_id = {process_id})"
            )
            self.send_message_init_config(
                process_id, global_model_params, process_id - 1
            )
            self.sender_id_to_client_index[process_id] = process_id - 1

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
            self.handle_message_receive_model_from_client,
        )

    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)

        self.aggregator.add_local_trained_result(
            sender_id - 1, model_params, local_sample_number
        )
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))
        if b_all_received:
            if self.args.method == "RFFL":
                reward_gradients = self.aggregator.aggregate(
                    self.sender_id_to_client_index
                )
            elif self.args.method == "QI":
                global_model_params = self.aggregator.aggregate(
                    self.sender_id_to_client_index, self.client_indexes
                )
            else:
                global_model_params = self.aggregator.aggregate(
                    self.sender_id_to_client_index
                )

            logging.info("Start Anomaly Detection")
            self.aggregator.anomalydetection(self.sender_id_to_client_index)
            self.aggregator.test_on_server_for_all_clients(self.round_idx)

            # start the next round
            self.round_idx += 1
            if self.round_idx == self.round_num:
                post_complete_message_to_sweep_process(self.args)
                self.finish()
                print("here")
                return
            if self.is_preprocessed:
                if self.preprocessed_client_lists is None:
                    # sampling has already been done in data preprocessor
                    self.client_indexes = [
                        self.round_idx
                    ] * self.args.client_num_per_round
                else:
                    self.client_indexes = self.preprocessed_client_lists[self.round_idx]
            else:
                # sampling clients
                self.client_indexes = self.aggregator.client_sampling(
                    self.round_idx,
                    self.args.client_num_in_total,
                    self.args.client_num_per_round,
                )

            print("indexes of clients: " + str(self.client_indexes))
            print("size = %d" % self.size)

            if self.args.method == "RFFL":
                for cid in self.client_indexes:
                    params = reward_gradients[cid]
                    if self.args.is_mobile == 1:
                        params = transform_grad_to_list(params)

                for process_id in range(1, self.size):
                    logging.info(
                        f"send updated parameters to client_index = {process_id - 1} (process_id = {process_id})"
                    )
                    self.send_message_sync_model_to_client(
                        process_id, params, process_id - 1
                    )
                    self.sender_id_to_client_index[process_id] = process_id - 1
            else:
                if self.args.is_mobile == 1:
                    global_model_params = transform_tensor_to_list(global_model_params)

                for process_id in range(1, self.size):
                    logging.info(
                        f"send updated parameters to client_index = {process_id - 1} (process_id = {process_id})"
                    )
                    self.send_message_sync_model_to_client(
                        process_id, global_model_params, process_id - 1
                    )
                    self.sender_id_to_client_index[process_id] = process_id - 1

    def send_message_init_config(self, receive_id, global_model_params, client_index):
        logging.info("Initial Configurations sent to client {0}".format(client_index))
        message = Message(
            MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id
        )
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)

    def send_message_sync_model_to_client(
        self, receive_id, global_model_params, client_index
    ):
        logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
            self.get_sender_id(),
            receive_id,
        )
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)