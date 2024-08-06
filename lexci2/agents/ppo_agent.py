"""Agent that uses Proximal Policy Optimization (PPO) for training.

File:   lexci2/agents/ppo_agent.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2022-04-21


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
from lexci2.agents.agent import lexci_logger_creator
from lexci2.agents.on_policy_agent import OnPolicyAgent
from lexci2.neural_network_modules.discrete_ppo_neural_network_module import (
    DiscretePpoNeuralNetworkModule,
)
from lexci2.neural_network_modules.continuous_ppo_neural_network_module import (
    ContinuousPpoNeuralNetworkModule,
)

import ray.rllib.agents.ppo as ppo
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.models.tf.tf_action_dist import Categorical, DiagGaussian
from ray.rllib.evaluation.postprocessing import compute_advantages
from keras.engine.functional import Functional
import tensorflow as tf

import copy
import numpy as np
from scipy.special import softmax
from typing import Any, Optional


class PpoAgent(OnPolicyAgent):
    """Agent that trains with PPO."""

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
            - trainer_config: dict[str, Any]
                  Configuration of the agent's trainer.
            - log_dir: str (default: '~/lexci_results/ray_results')
                  Folder to write the trainer's logs to.
        """

        super().__init__(id, env_config, trainer_config, log_dir)
        self._trainer = ppo.PPOTrainer(
            config=self._trainer_config,
            env=self._lexci_env_name,
            logger_creator=lexci_logger_creator(self._log_dir),
        )
        self._update_nn_module()

    def get_models(self) -> dict[str, Functional]:
        """Get all models of the agent, i.e. not only its policy NN but also
        value function approximators etc.

        Returns:
            - _: dict[str, Functional]:
                  A dictionary with all models of the agent.
        """

        model = self._trainer.get_policy().model.base_model
        return {"policy_vf_model": model}

    def set_models(self, new_models: dict[str, Functional]) -> None:
        """Set all models of the agent, i.e. not only its policy NN but also
        value function approximators etc.

        Arguments:
            - new_models: dict[str, Functional]
                  A dictionary containing the new models of the agent.

        Raises:
            - ValueError:
                  - If `models` is incomplete.
        """

        self._trainer.get_policy().model.base_model = new_models[
            "policy_vf_model"
        ]

    def _update_nn_module(self) -> None:
        """Update the neural network module.

        This method must be invoked after every training step.
        """

        # Create the new module
        if self._env_config.action_type == "discrete":
            self._nn_module = DiscretePpoNeuralNetworkModule(
                self._env_config, self.export_nn_to_bytes("keras"), "keras"
            )
        else:
            self._nn_module = ContinuousPpoNeuralNetworkModule(
                self._env_config, self.export_nn_to_bytes("keras"), "keras"
            )

    @staticmethod
    def get_default_trainer_config() -> dict[str, Any]:
        """Get the default configuration of the trainer.

        Returns:
            - _: dict[str, Any]
                  Default configuration.
        """

        return copy.deepcopy(ppo.DEFAULT_CONFIG)

    def get_nn(self) -> Functional:
        """Get the current policy neural network of the agent.

        This method may have to be overwritten for algorithms that don't store
        the policy NN in `base_model`.

        Returns:
            - _: Functional
                  Neural network of the agent.
        """

        return self.get_models()["policy_vf_model"]

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
            for exp in eps.exps:
                action = exp.action
                # RLlib seems to be able to handle NumPy arrays as continuous
                # actions, but not when the action space is discrete. In that
                # case, the action is passed as a `float`.
                if self._env_config.action_type == "discrete":
                    action = action[0]

                (
                    action_dist_input,
                    vf_pred,
                    action_logp,
                ) = self._calc_ppo_quantities(exp)
                action_prob = np.exp(action_logp)

                bb.add_values(
                    eps_id=i,
                    agent_index=eps.agent_id,
                    obs=exp.obs,
                    actions=action,
                    action_prob=action_prob,
                    action_logp=action_logp,
                    rewards=exp.reward,
                    dones=exp.done,
                    new_obs=exp.new_obs,
                    vf_preds=vf_pred,
                    action_dist_inputs=action_dist_input,
                )

            b = bb.build_and_reset()
            b = self._trainer.get_policy().postprocess_trajectory(b)
            episode_batches.append(b)

        for e in episode_batches:
            bb.add_batch(e)
        return bb.build_and_reset()

    def _calc_ppo_quantities(
        self, exp: Experience
    ) -> tuple[np.ndarray, float, float]:
        """Determine additional PPO-specific quantities.

        Arguments:
            - exp: Experience
                  Experience to calculate the quantities for.

        Returns:
            - action_dist_input: np.ndarray
                  An action distribution. If the action space is continuous,
                  this will be a NumPy array representing a Gaussian where the
                  first half contains the mean and the second half the natural
                  logarithm of the standard deviation.
            - vf_pred: float
                  Value function prediction.
            - action_logp: float
                  Log-likelihood of the chosen action.
        """

        action_dist_input = self._nn_module.get_norm_action_dist(exp.obs)
        vf_pred = self._nn_module.get_vf_pred(exp.obs)[0]

        params = copy.deepcopy(action_dist_input)
        params = np.reshape(params, (1, len(action_dist_input)))
        if self._env_config.action_type == "discrete":
            action_dist = Categorical(params, self._trainer.get_policy().model)
        else:
            action_dist = DiagGaussian(params, self._trainer.get_policy().model)

        action = copy.deepcopy(exp.action)
        if self._env_config.action_type == "continuous":
            action = np.reshape(action, (1, len(action)))
        action_logp = action_dist.logp(action).numpy()[0]

        return action_dist_input, vf_pred, action_logp
