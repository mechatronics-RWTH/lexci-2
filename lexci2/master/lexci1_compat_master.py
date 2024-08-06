"""An adaptation of the LExCI 2 Master which is compatible with LExCI 1 Minions.

File: lexci2/master/lexci1_compat_master.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2022-06-23


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

from lexci2.master.master import Master
from lexci2.minion.minion import Minion
from lexci2.communication.message import Message
from lexci2.communication.remote_procedure_call import (
    RemoteProcedureCall,
    DirectRemoteProcedureCall,
    BroadcastRemoteProcedureCall,
)
from lexci2.data_containers import Experience, Episode, Cycle
from lexci2.utils.csv_export import export_episode_csv

import os
import json
import time
import datetime
import copy
import logging
import numpy as np
import tensorflow as tf
from typing import Any, Union


logger = logging.getLogger(__name__)


class Lexci1CompatMaster(Master):
    """Adaptation of `Master` that is compatible with LExCI 1."""

    # Status constants
    FAILURE = 0
    SUCCESS = 1
    # Command constants
    START_MODEL_SOFTWARE = 0
    STOP_MODEL_SOFTWARE = 1
    START_TRAINING_RUN = 2
    START_VALIDATION_RUN = 3

    def _get_nn(self) -> list[Union[int, Any]]:
        """Get the neural network in a backwards-compatible format.

        To re-assemble Keras networks in the Minion, do the following:
            >>> import numpy as np
            >>> from tensorflow.keras.models import Model
            >>> nn_data = receive_nn_data()  # Data as returned by `_get_nn()`
            >>> config = nn_data[0]
            >>> weights = [np.array(e, dtype=np.float32) for e nn_data[1]]
            >>> nn = Model.from_config(config)
            >>> nn.set_weights(weights)  # Restored NN

        Returns:
            - _: list[Any]
                  Agent's neural network.
        """

        if self._nn_format == "keras":
            nn = self._agent._trainer.get_policy().model.base_model
            config = nn.get_config()
            config = json.dumps(config).replace(
                '"_initializer"', '"RandomNormal"'
            )
            config = json.loads(config)
            weights = nn.get_weights()
            nn_data = [config, [e.tolist() for e in weights]]
        else:
            nn_data = self._get_padded_nn_bytes_list()

        return nn_data

    def _start_model_software(self, minions: list[Minion]) -> list[Minion]:
        """Prompt Minions to start their model software.

        Arguments:
            - minions: list[Minion]
                  List of Minions.

        Returns:
            - _: list[Minion]
                  List of Minions that were able to start their model software.
        """

        # Command message to be sent to all connected minions
        cmd_msg = Message({"cmd": self.START_MODEL_SOFTWARE})

        # Send the broadcast RPC and wait for it to finish
        logger.info(
            f"Prompting {len(minions)} Minion(s) to start the model software."
        )
        rpc = BroadcastRemoteProcedureCall(
            minions, cmd_msg, self._minion_job_timeout
        )
        resp_msgs = rpc.get_result()

        # Check the status of the RPC.
        statuses = rpc.get_detailed_status()
        ready_minions = []
        for i in range(len(statuses)):
            if statuses[i] == RemoteProcedureCall.SUCCEEDED:
                try:
                    if (
                        resp_msgs[i].payload["ref"] == cmd_msg.id
                        and resp_msgs[i].payload["status"] == self.SUCCESS
                    ):
                        ready_minions.append(minions[i])
                except:
                    raise
            elif statuses[i] == RemoteProcedureCall.FAILED:
                minions[i].stop()

        return ready_minions

    def _mainloop(self) -> None:
        """Main loop of the master."""

        logger.info("Entering the main loop.")
        while True:
            # Request LExCI minions and wait until enough have connected
            self._listener.request_minions(
                self._max_num_minions - len(self._minions)
            )
            new_minions = self._listener.get_minions()
            new_minions = self._start_model_software(new_minions)
            self._minions.extend(new_minions)
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

            # Perform a validation run if this cycle calls for one
            if self._cycle_no % self._validation_interval == 0:
                # Save NN
                self._agent.save_checkpoint_file(self._checkpoint_dir)
                self._agent.export_nn_to_file(
                    os.path.join(self._nn_h5_dir, f"Cycle_{self._cycle_no}.h5"),
                    "keras",
                )
                # Run the validation
                try:
                    self._perform_validation_run()
                except:
                    logger.error(
                        "An error occurred during the validation run."
                        " Retrying..."
                    )
                    continue

            # Perform a training run
            try:
                self._perform_training_run()
            except:
                logger.error(
                    "An error occurred during the training run. Retrying..."
                )
                continue

            self._cycle_no += 1

    def _perform_validation_run(self) -> None:
        """Prompt a randomly chosen minion to perform a validation run.

        Raises:
            - RuntimeError
        """

        # Command message to be sent to one of the connected minions
        cmd_msg = Message(
            {
                "cmd": self.START_VALIDATION_RUN,
                "cycle": self._cycle_no,
                "nn_data": list(self._get_nn()),
                "minion_params": json.dumps(self._minion_params),
                "n_experiences": 1,
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
        cycle = Cycle()
        for eps in resp_msg.payload["episodes"]:
            episode = Episode(self._agent.get_id())
            for i in range(len(eps) - 1):
                exp = eps[i]
                next_exp = eps[i + 1]

                obs = np.array(exp["observation"], dtype=np.float32)
                action = np.array(exp["action"], dtype=np.float32)
                new_obs = np.array(next_exp["observation"], dtype=np.float32)
                reward = exp["reward"]
                done = False  # Only the last experience is marked as done
                t = exp["t"]
                aux_data = exp["aux_data"]

                experience = Experience(
                    obs, action, new_obs, reward, done, t, aux_data
                )
                episode.append_experience(experience)

                # LExCI 1's reinforcement learning block (i.e. the Simulink
                # block created for LExCI 1) doesn't store the new observation
                # in its experiences. Therefore, the last state is appended with
                # a neutral action and reward such that the required number of
                # experiences is collected.
                if i == len(eps) - 2:
                    obs = np.array(next_exp["observation"], dtype=np.float32)
                    action = 0 * np.array(next_exp["action"], dtype=np.float32)
                    new_obs = obs
                    reward = 0
                    done = True
                    t = next_exp["t"]
                    aux_data = next_exp["aux_data"]

                    experience = Experience(
                        obs, action, new_obs, reward, done, t, aux_data
                    )
                    episode.append_experience(experience)

            cycle.add_episode(episode)

        for i in range(len(cycle.eps)):
            file_name = f"Cycle_{self._cycle_no}_ValidationData_{i + 1}.csv"
            file_name = os.path.join(self._validation_data_dir, file_name)
            export_episode_csv(cycle.eps[i], file_name)

    def _perform_training_run(self) -> None:
        """Prompt all minions to perform generate training data.

        Raises:
            - RuntimeError
        """

        # Determine the number of experiences each minion has to gather
        num_cycle_exps = self._agent.get_num_experiences_per_cycle()
        num_experiences = int(np.ceil(num_cycle_exps / len(self._minions)))

        # Command message to be sent to one of the connected minions
        cmd_msg = Message(
            {
                "cmd": self.START_TRAINING_RUN,
                "cycle": self._cycle_no,
                "nn_data": list(self._get_nn()),
                "minion_params": json.dumps(self._minion_params),
                "n_experiences": num_experiences,
            }
        )

        # Send the broadcast RPC and wait for it to finish
        logger.info(
            f"Prompting {len(self._minions)} Minion(s) to perform a training"
            + " run."
        )
        rpc = BroadcastRemoteProcedureCall(
            self._minions, cmd_msg, self._minion_job_timeout
        )
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
            for eps in msg.payload["episodes"]:
                episode = Episode(self._agent.get_id())
                for i in range(len(eps) - 1):
                    exp = eps[i]
                    next_exp = eps[i + 1]

                    obs = np.array(exp["observation"], dtype=np.float32)
                    action = np.array(exp["action"], dtype=np.float32)
                    new_obs = np.array(
                        next_exp["observation"], dtype=np.float32
                    )
                    reward = exp["reward"]
                    done = False  # Only the last experience is marked as done
                    t = exp["t"]
                    aux_data = exp["aux_data"]

                    experience = Experience(
                        obs, action, new_obs, reward, done, t, aux_data
                    )
                    episode.append_experience(experience)

                    # LExCI 1's reinforcement learning block (i.e. the Simulink
                    # block created for LExCI 1) doesn't store the new
                    # observation in its experiences. Therefore, the last state
                    # is appended with a neutral action and reward such that the
                    # required number of experiences is collected.
                    if i == len(eps) - 2:
                        obs = np.array(
                            next_exp["observation"], dtype=np.float32
                        )
                        action = 0 * np.array(
                            next_exp["action"], dtype=np.float32
                        )
                        new_obs = obs
                        reward = 0
                        done = True
                        t = next_exp["t"]
                        aux_data = next_exp["aux_data"]

                        experience = Experience(
                            obs, action, new_obs, reward, done, t, aux_data
                        )
                        episode.append_experience(experience)

                aggregated_cycle.add_episode(episode)

        # Save the data
        if self._save_training_data:
            for i in range(len(aggregated_cycle.eps)):
                file_name = f"Cycle_{self._cycle_no}_TrainingData_{i + 1}.csv"
                file_name = os.path.join(self._training_data_dir, file_name)
                export_episode_csv(aggregated_cycle.eps[i], file_name)

        # Log the metrics
        self._csv_logger.write(aggregated_cycle, self._cycle_no)

        logger.info("Training the agent...")
        t_start = datetime.datetime.now()
        self._agent.train(aggregated_cycle)
        t_end = datetime.datetime.now()
        t_training = (t_end - t_start).total_seconds()
        logger.info(f"Training finished in {t_training:.1f} s.")
