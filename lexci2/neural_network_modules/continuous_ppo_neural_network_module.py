"""This file contains a helper class for interacting with a continuous PPO
agent's neural network. It assumes that the underlying `NeuralNetwork` has only
a single input layer. The number of output layers is not limited though.

File:   lexci2/neural_network_modules/continuous_ppo_neural_network_module.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2023-03-03


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


from lexci2.neural_network_modules.neural_network_module import (
    NeuralNetworkModule,
)
from lexci2.lexci_env import LexciEnvConfig
from lexci2.utils.transform import transform_linear

import numpy as np


class ContinuousPpoNeuralNetworkModule(NeuralNetworkModule):
    """Helper class for handling an agent's neural network.

    It presumes that the NN only has a single input layer.
    """

    def __init__(
        self,
        env_config: LexciEnvConfig,
        nn_data: bytes,
        nn_data_fmt: str = "tflite",
        **kwargs
    ) -> None:
        """Initialize the neural network module.

        Arguments:
            - env_config: LexciEnvConfig
                  Configuration of the LExCI environment, i.e. the dimensions,
                  bounds, and types of the observation and action space.
            - nn_data: bytes
                  Data of the neural network, i.e. its weights, biases, etc.
            - nn_data_fmt: str (Default: "tflite")
                  Format of the neural network data (must be either "keras" or
                  "tflite").
            - kwargs
                  Optional keyword arguments.
                  - tensor_arena_size: int (Default: 1000000, Unit: B)
                        Size of the tensor arena. This argument is ignored if
                        the neural network data format is "keras".

        Raises:
            - ValueError
        """

        super().__init__(env_config, nn_data, nn_data_fmt, **kwargs)

    def get_norm_action_dist(
        self, norm_obs: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Get the agent's normalized action distribution for a normalized
        observation.

        Arguments:
            - norm_obs: np.ndarray
                  A normalized observation.
            - kwargs
                  Optional keyword arguments.

        Returns:
            - _: np.ndarray
                  The agent's normalized action distribution.
        """

        if self._nn_data_fmt == "keras":
            return self.predict(norm_obs)[0][0]
        elif self._nn_data_fmt == "tflite":
            return self.predict(norm_obs)[0]

    def get_norm_action(
        self, norm_obs: np.ndarray, b_sample: bool = True, **kwargs
    ) -> np.ndarray:
        """Get a normalized action for a given observation.

        Actions are sampled from the action distribution that the neural network
        yields for the normalized observation. If sampling is deactivated, the
        mean or the most likely action is returned.

        Arguments:
            - norm_obs: np.ndarray:
                  A normalized observation.
            - b_sample: bool (Default: True)
                  If true, the action is sampled from the action distribution.
                  Otherwise, the mean of the distribution is returned.
            - kwargs
                  Optional keyword arguments.

        Returns:
            - _: np.ndarray
                  A normalized action.
        """

        norm_action_dist = self.get_norm_action_dist(norm_obs)
        mean, stdev = np.split(norm_action_dist, 2)
        if b_sample:
            norm_action = mean + np.random.normal(size=stdev.shape) * np.exp(
                stdev
            )
        else:
            norm_action = mean

        return norm_action

    def get_vf_pred(self, norm_obs: np.ndarray) -> np.ndarray:
        """Get the predicted value function value.

        Arguments:
            - norm_obs: np.ndarray
                  A normalized observation.

        Returns:
            - _: np.ndarray
                  The predicted value of the value function approximator.
        """

        if self._nn_data_fmt == "keras":
            return self.predict(norm_obs)[1][0]
        elif self._nn_data_fmt == "tflite":
            return self.predict(norm_obs)[1]

    def normalize_action_dist(self, action_dist: np.ndarray) -> np.ndarray:
        """Normalize an action distribution.

        Arguments:
            - action_dist: np.ndarray
                  An action distribution.

        Returns:
            - _: np.ndarray
                  The normalized action distribution.
        """

        # Scaling factor between the image of `tanh`, i.e. [-1, +1], and the
        # action space of the environment
        s = (self._env_config.action_ub - self._env_config.action_lb) / 2

        means, stdevs = np.split(action_dist, 2)

        norm_means = self.normalize_action(means)
        norm_stdevs = np.log(np.arctanh(stdevs / s))

        return np.concatenate((norm_means, norm_stdevs))

    def denormalize_action_dist(
        self, norm_action_dist: np.ndarray
    ) -> np.ndarray:
        """Denormalize an action distribution.

        Arguments:
            - norm_action_dist: np.ndarray
                  A normalized action distribution.

        Returns:
            - _: np.ndarray
                  The denormalized action distribution.
        """

        # Scaling factor between the image of `tanh`, i.e. [-1, +1], and the
        # action space of the environment
        s = (self._env_config.action_ub - self._env_config.action_lb) / 2

        norm_means, norm_stdevs = np.split(norm_action_dist, 2)

        means = self.denormalize_action(norm_means)
        stdevs = s * np.tanh(np.exp(norm_stdevs))

        return np.concatenate((means, stdevs))
