"""Agent that utilizes Deep Deterministic Policy Gradient (DDPG) for training.

File:   lexci2/agents/ddpg_agent.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2022-07-18


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


from lexci2.lexci_env import LexciEnvConfig
from lexci2.data_containers import Experience, Cycle
from lexci2.agents.agent import lexci_logger_creator, nn_modifying_method
from lexci2.agents.off_policy_agent import OffPolicyAgent
from lexci2.neural_network_modules.ddpg_neural_network_module import (
    DdpgNeuralNetworkModule,
)

import ray.rllib.agents.ddpg as ddpg
from ray.rllib.agents.ddpg.lexci_ddpg import LexciDdpgTrainer
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
import tensorflow as tf
import tensorflow.keras
from keras.engine.functional import Functional

import copy
import numpy as np
from typing import Any, Optional


class DdpgAgent(OffPolicyAgent):
    """Agent that trains with DDPG."""

    def __init__(
        self,
        id: str,
        env_config: LexciEnvConfig,
        trainer_config: dict[str, Any],
        log_dir: str = "~/lexci_results/ray_results",
    ) -> None:
        """Initialize the agent.

        Arguments:
            - id: str
                  ID of the agent.
            - env_config: LexciEnvConfig
                  Configuration of the agent's environment.
            - trainer_config: Dict[str, Any]
                  Configuration of the agent's trainer.
            - log_dir: str (default: '~/lexci_results/ray_results')
                  Folder to write the trainer's log to.
        """

        super().__init__(id, env_config, trainer_config, log_dir)

        self._trainer_config["use_state_preprocessor"] = False
        self._trainer_config["_disable_execution_plan_api"] = True
        self._trainer_config["timesteps_per_iteration"] = 0
        self._trainer_config["learning_starts"] = self._trainer_config[
            "train_batch_size"
        ]
        self._trainer = LexciDdpgTrainer(
            config=self._trainer_config,
            env=self._lexci_env_name,
            logger_creator=lexci_logger_creator(self._log_dir),
        )
        self._update_nn_module()

    def get_nn(self) -> Functional:
        """Get the current policy neural network of the agent.

        This method may have to be overwritten for algorithms that don't store
        the policy NN in `base_model`.

        Returns:
            - _: Functional
                  Neural network of the agent.
        """

        return self._trainer.get_policy().model.policy_model

    def _update_nn_module(self) -> None:
        """Update the neural network module.

        This method must be invoked after every training step.
        """

        # Create the new module
        self._nn_module = DdpgNeuralNetworkModule(
            self._env_config, self.export_nn_to_bytes("keras"), "keras"
        )

    @staticmethod
    def get_default_trainer_config() -> dict[str, Any]:
        """Get the default configuration of the trainer.

        Returns:
            - _: dict[str, Any]:
                  Default configuration.
        """

        return copy.deepcopy(ddpg.DEFAULT_CONFIG)

    def import_model_h5(self, model_h5_file: str) -> None:
        """Import a model (i.e. a TensorFlow neural network) from an h5-file and
        overwrite the trainer's model with it.

        TODO: Which network is imported here? The actor, the critic, or both?

        Arguments:
          - model_h5_file: str
              Path to the model h5-file to import.
        """

        # TODO
        raise NotImplementedError

    def _create_batch(self, cycle: Cycle) -> SampleBatch:
        """Postprocess cycle data and convert it into a `SampleBatch`.

        Arguments:
            - cycle: Cycle
                  Training cycle data.

        Returns:
            - _: SampleBatch
                  Preprocessed training cycle data.
        """

        bb = SampleBatchBuilder()
        episode_batches = []

        for i, eps in enumerate(cycle.eps):
            prev_action = None
            prev_reward = None

            for exp in eps.exps:
                action = exp.action
                # RLlib seems to be able to handle NumPy arrays as continuous
                # actions, but not when the action space is discrete. In that
                # case, the action is passed as a `float`.
                if self._env_config.action_type == "discrete":
                    action = action[0]

                # For the very first experience, set the previous action and
                # reward to the current values
                if prev_action is None:
                    prev_action = action
                if prev_reward is None:
                    prev_reward = exp.reward

                action_dist_input = self._nn_module.get_norm_action_dist(
                    exp.obs
                )

                bb.add_values(
                    eps_id=i,
                    agent_index=eps.agent_id,
                    obs=exp.obs,
                    actions=action,
                    prev_actions=prev_action,
                    rewards=exp.reward,
                    prev_rewards=prev_reward,
                    dones=exp.done,
                    new_obs=exp.new_obs,
                    action_dist_inputs=action_dist_input,
                )

                prev_action = action
                prev_reward = exp.reward

            b = bb.build_and_reset()
            b = self._trainer.get_policy().postprocess_trajectory(b)
            episode_batches.append(b)

        for e in episode_batches:
            bb.add_batch(e)
        return bb.build_and_reset()

    @nn_modifying_method
    def train_on_cycle(
        self, cycle: Cycle, b_add_to_memory: bool = True
    ) -> None:
        """Train on experiences from a given cycle.

        This method allows the agent to train on on-policy data only, i.e. on
        experiences that were collected using the latest version of the policy
        network.
        Implementations must invoke `self._update_nn_module()` after training.

        Arguments:
            - cycle: Cycle
                  A LExCI cycle.
            - b_add_to_memory: bool (default: True)
                  Whether the experiences shall be added to the agent's replay
                  memory after training is done.
        """

        batch = self._create_batch(cycle)
        self._trainer.train_on_given_batch(batch, b_add_to_memory)
        self._update_nn_module()

    @nn_modifying_method
    def train_on_replay_memory(self) -> None:
        """Train using only experiences from the replay memory.

        Implementations must invoke `self._update_nn_module()` after training.
        """

        self._trainer.train_on_replay_memory()
        self._update_nn_module()

    def add_lexci_cycle_to_replay_memory(self, cycle: Cycle) -> None:
        """Add experiences to the agent's replay memory.

        Arguments:
            - cycle: Cycle
                  A LExCI cycle containing the experiences that shall be added
                  to the replay buffer of the agent.
        """

        batch = self._create_batch(cycle)
        self._trainer.add_to_replay_memory(batch)

    def add_sample_batch_to_replay_memory(
        self, sample_batch: SampleBatch
    ) -> None:
        """Add experiences to the agent's replay memory.

        Arguments:
            - sample_batch: SampleBatch
                  A `SampleBatch` containing the data that shall be added to the
                  replay buffer of the agent.
        """

        self._trainer.add_to_replay_memory(sample_batch)

    def get_replay_memory_size(self) -> int:
        """Get the number of experiences in the replay memory buffer.

        Returns:
            - _: int
                  Number of experiences in the replay memory.
        """

        return self._trainer.get_replay_memory_size()
