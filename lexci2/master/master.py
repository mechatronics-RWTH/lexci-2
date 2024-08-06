"""Master of the (L)earning and (Ex)periencing (C)ycle (I)nterface (LExCI).

File:   master/master.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2022-05-20


Copyright 2023 Teaching and Research Area Mechatronics in Mobile Propulsion,
               RWTH Aachen University

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at: http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

from lexci2.agents.agent import Agent
from lexci2.agents.off_policy_agent import OffPolicyAgent
from lexci2.master.listener import Listener
from lexci2.communication.message import Message
from lexci2.communication.remote_procedure_call import (
    RemoteProcedureCall,
    DirectRemoteProcedureCall,
    BroadcastRemoteProcedureCall,
)
from lexci2.data_containers import Cycle
from lexci2.utils.csv_export import export_episode_csv
from lexci2.utils.offline_data import (
    import_sample_batch_json_folder,
    export_sample_batch_json,
)
from lexci2.utils.csv_logger import CsvLogger

import os
import time
import datetime
import copy
import logging
import re
import numpy as np
from typing import Any


logger = logging.getLogger(__name__)


class Master:
    """Server that communicates with LExCI Minions in order to gather
    experiences and train its agent.
    """

    def __init__(
        self,
        agent: Agent,
        addr: str,
        port: int,
        *,
        num_experiences_per_cycle: int = 0,
        mailbox_buffer_size: int = 1 * 1024**3,
        min_num_minions: int = 1,
        max_num_minions: int = 1,
        minion_job_timeout: float = 3600,
        minion_params: dict[str, Any] = {},
        nn_format: str = "keras",
        nn_size: int = 1 * 1024**2,
        output_dir: str = "~/lexci_results",
        b_save_training_data: bool = False,
        b_save_sample_batches: bool = False,
        validation_interval: int = 10,
        num_replay_trainings: int = 0,
        perc_replay_trainings: float = 0,
        num_exp_before_replay_training: int = 0,
        offline_data_import_folder: str = "",
        b_offline_training_only: bool = False,
    ) -> None:
        """Initialize the Master.

        Arguments:
            - agent: Agent
                  Agent to be used and trained by the Master.
            - addr: str
                  IP-address of the server.
            - port: int
                  Listening port.
            - num_experiences_per_cycle: int (Default: 0)
                  Number of experiences to generate per LExCI cycle. If this is
                  less than what the agent requires for training, it is set to
                  the agent's value.
            - mailbox_buffer_size: int (Default: 1 * 1024**3, Unit: B)
                  Buffer size of each `Mailbox`.
            - min_num_minions: int (Default: 1)
                  Minimum number of LExCI Minions that must be connected to the
                  Master in order to run.
            - max_num_minions: int (Default: 1)
                  Maximum number of LExCI Minions that can be connected to the
                  Master at the same time.
            - minion_job_timeout: float (Default: 3600, Unit: s)
                  Timeout for jobs performed by Minions.
            - minion_params: dict[str, Any] (Default: {})
                  Dictionary containing additional parameters that are sent
                  every cycle to the Minions.
            - nn_format: str (Default: 'keras')
                  Format of the neural network bytes that are sent to the
                  Minions. This must be either 'keras' or 'tflite'.
            - nn_size: int (Default: 1 * 1024**2, Unit: B)
                  Size of the padded neural network bytes.
            - output_dir: str (Default: '~/lexci_results')
                  Directory where logs, checkpoints, validation data, etc. are
                  stored.
            - b_save_training_data: bool (Default: False)
                  Whether to save training cycle data as CSV files.
            - b_save_sample_batches: bool (Default: False)
                  Whether to save the generated sample batches as JSON files.
            - validation_interval: int (Default: 10)
                  Number of LExCI cycles between validation runs.
            - num_replay_trainings: int (Default: 0)
                  Number of off-policy trainings using the replay memory after
                  every on-policy training. If the agent doesn't use an
                  off-policy algorithm, this is always set to 0. This parameter
                  (if not 0) takes precedence over `perc_replay_trainings`.
            - perc_replay_trainings: float (Default: 0)
                  Percentage of the replay buffer that shall be used for offline
                  training. This parameter is ignored if `num_replay_trainings`
                  is greater than zero.
            - num_exp_before_replay_training: int (Default: 0)
                  Do not start training on the replay buffer before this many
                  experiences have been collected. If it's smaller than
                  `num_experiences_per_cycle`, it's set to that value.
            - offline_data_import_folder: str (Default: "")
                  Folder to import offline data from. The files in the directory
                  must be sample batch JSONs. If the string is empty, this
                  feature is disabled.
            - b_offline_training_only: bool (Default: False)
                  Whether to traing exclusively on offline data. If this is set,
                  the Master doesn't listen for Minions.
        """

        # Timestamp of when the Master object was created
        s = str(datetime.datetime.now()).replace(" ", "_").replace(":", "-")
        self._init_time_str = s

        # Copy values
        self._agent = agent
        self._addr = addr
        self._port = port
        self._num_experiences_per_cycle = num_experiences_per_cycle
        self._mailbox_buffer_size = mailbox_buffer_size
        self._min_num_minions = min_num_minions
        self._max_num_minions = max_num_minions
        self._minion_job_timeout = minion_job_timeout
        self._minion_params = copy.deepcopy(minion_params)
        self._nn_format = nn_format
        self._nn_size = nn_size
        self._output_dir = output_dir
        self._b_save_training_data = b_save_training_data
        self._b_save_sample_batches = b_save_sample_batches
        self._validation_interval = validation_interval
        self._num_replay_trainings = num_replay_trainings
        self._perc_replay_trainings = perc_replay_trainings
        self._num_exp_before_replay_training = num_exp_before_replay_training
        self._offline_data_import_folder = offline_data_import_folder
        self._b_offline_training_only = b_offline_training_only

        # Correct the number of experiences per cycle if needed
        if (
            self._num_experiences_per_cycle
            < self._agent.get_num_training_experiences()
        ):
            self._num_experiences_per_cycle = (
                self._agent.get_num_training_experiences()
            )

        # Correct the minimum replay buffer size for offline training if needed
        if (
            self._num_exp_before_replay_training
            < self._num_experiences_per_cycle
        ):
            self._num_exp_before_replay_training = (
                self._num_experiences_per_cycle
            )

        # Ensure that training on the replay memory is deactivated for on-policy
        # agents
        if self._num_replay_trainings > 0 and not isinstance(
            self._agent, OffPolicyAgent
        ):
            logger.warn(
                "The agent isn't based on an off-policy algorithm. Setting the"
                + " number of off-policy trainings to 0."
            )
            self._num_replay_trainings = 0
        if self._perc_replay_trainings > 0 and not isinstance(
            self._agent, OffPolicyAgent
        ):
            logger.warn(
                "The agent isn't based on an off-policy algorithm. Setting the"
                + " percentage of the replay buffer that shall be used for"
                + " offline training to 0."
            )
            self._perc_replay_trainings = 0

        # Disable importing offline data for on-policy agents
        if self._offline_data_import_folder != "" and not isinstance(
            self._agent, OffPolicyAgent
        ):
            logger.warn(
                "The agent isn't based on an off-policy algorithm. Deactivating"
                + " the import of off-policy data."
            )
            self._offline_data_import_folder = ""

        # Turn off training exclusively on offline data for on-policy agents
        if self._b_offline_training_only and not isinstance(
            self._agent, OffPolicyAgent
        ):
            logger.warn(
                "The agent isn't based on an off-policy algorithm. Deactivating"
                + " training exclusively on off-policy data."
            )
            self._b_offline_training_only = False

        # Set the minimum and number of Minions to 0 when training on off-policy
        # data only
        if self._b_offline_training_only:
            self._min_num_minions = 0
            self._max_num_minions = 0

        # Communication and logging
        self._minions = []
        self._listener = Listener(
            self._addr, self._port, self._mailbox_buffer_size
        )
        self._cycle_no = 0
        self._csv_logger = None

        # Create a directory with the current timestamp inside the output folder
        self._output_dir = os.path.expanduser(self._output_dir)
        self._output_dir = os.path.join(self._output_dir, self._init_time_str)
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)

        # Create sub-folder for the trainer's logs (i.e. 'ray_results')
        self._ray_results_dir = os.path.join(self._output_dir, "ray_results")
        if not os.path.exists(self._ray_results_dir):
            os.makedirs(self._ray_results_dir)
        self._agent.set_log_dir(self._ray_results_dir)

        # Create sub-folder for checkpoints
        self._checkpoint_dir = os.path.join(self._output_dir, "Checkpoints")
        if not os.path.exists(self._checkpoint_dir):
            os.makedirs(self._checkpoint_dir)

        # Create sub-folder for h5-files
        self._nn_h5_dir = os.path.join(self._output_dir, "NN_h5")
        if not os.path.exists(self._nn_h5_dir):
            os.makedirs(self._nn_h5_dir)

        # Create sub-folder for sample batch JSONs
        self._sample_batch_json_dir = os.path.join(
            self._output_dir, "Sample_Batch_JSONs"
        )
        if not os.path.exists(self._sample_batch_json_dir):
            os.makedirs(self._sample_batch_json_dir)

        # Create sub-folder for training data
        self._training_data_dir = os.path.join(
            self._output_dir, "Training_Data"
        )
        if not os.path.exists(self._training_data_dir):
            os.makedirs(self._training_data_dir)

        # Create sub-folder for validation data
        self._validation_data_dir = os.path.join(
            self._output_dir, "Validation_Data"
        )
        if not os.path.exists(self._validation_data_dir):
            os.makedirs(self._validation_data_dir)

    def start(self) -> None:
        """Start the Master."""

        logger.info("LExCI Master started.")
        self._csv_logger = CsvLogger(os.path.join(self._output_dir, "log.csv"))
        self._listener.start()
        self._mainloop()

    def stop(self) -> None:
        """Stop the Master."""

        self._listener.stop()
        if self._csv_logger is not None:
            self._csv_logger.close()
            self._csv_logger = None
        logger.info("LExCI Master stopped.")

    def restore_checkpoint(self, checkpoint_file: str) -> None:
        """Load a checkpoint.

        Arguments:
            - checkpoint_file: str
                  Path to the checkpoint file to restore.
        """

        # Load the checkpoint
        self._agent.load_checkpoint_file(checkpoint_file)
        logger.info(f"Restored checkpoint '{checkpoint_file}'.")

        # Set the cycle number to the one of the checkpoint
        m = re.match(
            "^checkpoint-(?P<cycle_no>\d+)$", os.path.basename(checkpoint_file)
        )
        self._cycle_no = int(m["cycle_no"])

    def import_models_h5(self, model_h5_folder_name: str) -> None:
        """Import models (i.e. TensorFlow neural networks) from h5-files and
        overwrite the agent's models with them.

        Arguments:
            - model_h5_folder_name: str
                  Path to the folder containing the h5-files to import.
        """

        self._agent.import_models(model_h5_folder_name)
        logger.info(f"Imported model from '{model_h5_folder_name}'.")

    def _mainloop(self) -> None:
        """Main loop of the Master."""

        # Import offline data
        if self._offline_data_import_folder != "":
            logger.info(
                "Importing offline data JSONs from"
                + f" '{self._offline_data_import_folder}'."
            )
            imported_sample_batches = import_sample_batch_json_folder(
                self._offline_data_import_folder
            )
            for i, e in enumerate(imported_sample_batches):
                self._agent.add_sample_batch_to_replay_memory(e)
                logger.info(
                    f"Wrote {i + 1}/{len(imported_sample_batches)} imported"
                    " sample batches to the replay buffer."
                )

        logger.info("Entering the main loop.")
        while True:
            # Request LExCI minions and wait until enough have connected
            self._listener.request_minions(
                self._max_num_minions - len(self._minions)
            )
            self._minions.extend(self._listener.get_minions())
            if len(self._minions) < self._min_num_minions:
                logger.info(
                    "Waiting for the minimum number of Minions to connect."
                    + " Currently,"
                    + f" {len(self._minions)}/{self._min_num_minions} are"
                    + " available."
                )
                time.sleep(1.0)
                continue

            logger.info(f"Starting cycle {self._cycle_no}.")

            # Explicitly set the iteration number. This is necessarity so that
            # it corresponds with the cycle number when there are multiple
            # training steps (e.g. when using the DDPG agent).
            self._agent.set_cycle_no(self._cycle_no)

            # Perform a validation run if this cycle calls for one
            if self._cycle_no % self._validation_interval == 0:
                # Save NN
                self._agent.save_checkpoint_file(self._checkpoint_dir)
                self._agent.export_models(
                    os.path.join(self._nn_h5_dir, f"Cycle_{self._cycle_no}")
                )
                # Run the validation
                try:
                    self._perform_validation_run()
                except Exception as e:
                    logger.error(f"Caught the following exception: {e}")
                    logger.error(
                        "An error occurred during the validation run."
                        " Retrying..."
                    )
                    continue

            # Perform a training run
            try:
                self._perform_training_run()
            except Exception as e:
                logger.error(f"Caught the following exception: {e}")
                logger.error(
                    "An error occurred during the training run. Retrying..."
                )
                continue

            self._cycle_no += 1

    def _get_padded_nn_bytes_list(self) -> list[int]:
        """Get the padded bytes of the agent's neural network.

        Returns:
            - _: list[int]:
                  Bytes of the agent's neural network, padded to `self._nn_size`
                  with zeros, as a list of integers.
        """

        nn_bytes_list = list(self._agent.export_nn_to_bytes(self._nn_format))
        nn_size = len(nn_bytes_list)
        if nn_size < self._nn_size:
            nn_bytes_list.extend((self._nn_size - nn_size) * [0])
        return nn_bytes_list

    def _perform_validation_run(self) -> None:
        """Prompt a randomly chosen minion to perform a validation run.

        Raises:
            - RuntimeError
        """

        # There's no Minion to perform a validation run when training on offline
        # data ony
        if self._b_offline_training_only:
            return

        # Command message to be sent to one of the connected minions
        cmd_msg = Message(
            {
                "cmd": "perform_validation_run",
                "cycle_no": self._cycle_no,
                "nn_bytes": self._get_padded_nn_bytes_list(),
                "minion_params": self._minion_params,
            }
        )

        # Randomly select one of the connected minions and ask it to perform a
        # validation run
        idx = np.random.randint(0, len(self._minions))
        minion_addr, minion_port = self._minions[idx].get_peer_addr()
        logger.info(
            f"Prompting Minion {minion_addr}:{minion_port} to perform a"
            + " validation run."
        )
        rpc = DirectRemoteProcedureCall(
            self._minions[idx], cmd_msg, self._minion_job_timeout
        )

        # Wait for the RPC to finish
        while rpc.get_status() == RemoteProcedureCall.RUNNING:
            time.sleep(0.1)
        if rpc.get_status() == RemoteProcedureCall.FAILED:
            logger.warn(
                f"Minion {minion_addr}:{minion_port} failed. Closing the"
                + " connection."
            )
            self._minions[idx].stop()
            del self._minions[idx]
            raise RuntimeError("Validation run failed.")
        logger.info("Validation run completed.")
        resp_msg = rpc.get_result()

        # Save the data
        cycle = Cycle.from_json(resp_msg.payload["cycle_json"])
        for i in range(len(cycle.eps)):
            file_name = f"Cycle_{self._cycle_no}_ValidationData_{i + 1}.csv"
            file_name = os.path.join(self._validation_data_dir, file_name)
            export_episode_csv(cycle.eps[i], file_name)
        if self._b_save_sample_batches:
            file_name = f"Cycle_{self._cycle_no}_Validation_SampleBatch.json"
            file_name = os.path.join(self._sample_batch_json_dir, file_name)
            sample_batch = self._agent._create_batch(cycle)
            export_sample_batch_json(sample_batch, file_name)

    def _perform_training_run(self) -> None:
        """Prompt all minions to perform generate training data.

        Raises:
            - RuntimeError
        """

        if not self._b_offline_training_only:
            # Determine the number of experiences each minion has to gather
            num_cycle_exps = self._num_experiences_per_cycle
            num_experiences = int(np.ceil(num_cycle_exps / len(self._minions)))

            # Command message to be sent to all the connected minions
            cmd_msg = Message(
                {
                    "cmd": "perform_training_run",
                    "cycle_no": self._cycle_no,
                    "nn_bytes": self._get_padded_nn_bytes_list(),
                    "minion_params": self._minion_params,
                    "num_experiences": num_experiences,
                }
            )

            # Send the broadcast RPC
            logger.info(
                f"Prompting {len(self._minions)} Minion(s) to perform a"
                + " training run."
            )
            rpc = BroadcastRemoteProcedureCall(
                self._minions, cmd_msg, self._minion_job_timeout
            )

        # Train on the replay memory if requested and possible
        if self._num_replay_trainings > 0:
            num_replay_trainings = self._num_replay_trainings
        elif self._perc_replay_trainings > 0:
            num_replay_trainings = int(
                self._perc_replay_trainings
                * self._agent.get_replay_memory_size()
                / self._agent.get_num_training_experiences()
            )
        else:
            num_replay_trainings = 0
        for _ in range(num_replay_trainings):
            if (
                self._agent.get_replay_memory_size()
                >= self._num_exp_before_replay_training
            ):
                self._agent.train_on_replay_memory()

        if not self._b_offline_training_only:
            # Wait for the RPC to finish
            resp_msgs = rpc.get_result()

            # Check the status of the RPC. If it failed, remove all minions that
            # weren't successful.
            if rpc.get_status() == RemoteProcedureCall.FAILED:
                statuses = rpc.get_detailed_status()
                minions = self._minions.copy()
                self._minions.clear()
                for i in range(len(statuses)):
                    if statuses[i] == RemoteProcedureCall.SUCCEEDED:
                        self._minions.append(minions[i])
                    elif statuses[i] == RemoteProcedureCall.FAILED:
                        minions[i].stop()

                raise RuntimeError("Training run failed.")
            logger.info("Training run completed.")

            # Aggregate response messages into a single `Cycle` to train on
            aggregated_cycle = Cycle()
            for msg in resp_msgs:
                minion_cycle = Cycle.from_json(msg.payload["cycle_json"])
                for eps in minion_cycle.eps:
                    aggregated_cycle.add_episode(eps)

            # Save the data
            if self._b_save_training_data:
                for i in range(len(aggregated_cycle.eps)):
                    file_name = (
                        f"Cycle_{self._cycle_no}_TrainingData_{i + 1}.csv"
                    )
                    file_name = os.path.join(self._training_data_dir, file_name)
                    export_episode_csv(aggregated_cycle.eps[i], file_name)
            if self._b_save_sample_batches:
                file_name = f"Cycle_{self._cycle_no}_Training_SampleBatch.json"
                file_name = os.path.join(self._sample_batch_json_dir, file_name)
                sample_batch = self._agent._create_batch(aggregated_cycle)
                export_sample_batch_json(sample_batch, file_name)

            # Log the metrics
            self._csv_logger.write(aggregated_cycle, self._cycle_no)

            logger.info("Training the agent...")
            t_start = datetime.datetime.now()
            self._agent.train(aggregated_cycle)
            t_end = datetime.datetime.now()
            t_training = (t_end - t_start).total_seconds()
            logger.info(f"Training finished in {t_training:.1f} s.")

    def get_output_dir(self) -> str:
        """Get the output directory of the current training.

        Returns:
            - _: str
                  Absolute path to the output directory of the current training.
        """

        return self._output_dir
