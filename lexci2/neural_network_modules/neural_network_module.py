"""This file contains a helper class for interacting with an agent's neural
network. It assumes that the underlying `NeuralNetwork` has only a single input
layer. The number of output layers is not limited though.

File:   lexci2/neural_network_modules/neural_network_module.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2023-03-01


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
from lexci2.neural_network_modules.neural_networks.keras_neural_network import (
    KerasNeuralNetwork,
)
from lexci2.neural_network_modules.neural_networks.tflite_neural_network import (
    TfliteNeuralNetwork,
)
from lexci2.utils.transform import (
    transform_linear,
    transform_tanh,
    transform_atanh,
)

from abc import ABCMeta, abstractmethod
import copy
import numpy as np
import logging


# Logger
logger = logging.getLogger(__name__)


class NeuralNetworkModule(metaclass=ABCMeta):
    """Helper class for handling an agent's neural network.

    It presumes that the NN only has a single input layer.
    """

    def __init__(
        self,
        env_config: LexciEnvConfig,
        nn_data: bytes,
        nn_data_fmt: str = "tflite",
        *,
        discrete_action_map: dict[int, np.ndarray] = None,
        **kwargs,
    ) -> None:
        """Initialize the neural network module.

        Arguments:
            - env_config: LexciEnvConfig
                  Configuration of the LExCI environment, i.e. the dimensions,
                  bounds, and types of the observation and action space.
            - nn_data: bytes
                  Data of the neural network, i.e. its weights, biases, etc.
            - nn_data_fmt: str (Default: 'tflite')
                  Format of the neural network data (must be either 'keras' or
                  'tflite').
            - discrete_action_map: dict[int, np.ndarray] (Default: None)
                  Bijective mapping from normalized, discrete actions to
                  denormalized actions, i.e. for denormalizing. This is used by
                  the `normalize_discrete_action()` and
                  `denormalize_discrete_action()` method.
            - kwargs
                  Optional keyword arguments.
                      - tensor_arena_size: int (Default: 1000000, Unit: B)
                            Size of the tensor arena. This argument is ignored
                            if the neural network data format is "keras".

        Raises:
            - ValueError
        """

        self._env_config = copy.deepcopy(env_config)
        self._nn = None
        self._nn_data_fmt = nn_data_fmt

        if self._nn_data_fmt == "keras":
            self._nn = KerasNeuralNetwork(nn_data)
        elif self._nn_data_fmt == "tflite":
            tensor_arena_size = kwargs.get("tensor_arena_size", 1000000)
            self._nn = TfliteNeuralNetwork(nn_data, tensor_arena_size)
        else:
            raise ValueError(
                "Unknown neural network data format" f" '{self._nn_data_fmt}'."
            )

        # Mapping used for (de-)normalizing discrete actions
        self._discrete_action_map = discrete_action_map

    def predict(self, norm_obs: np.ndarray, **kwargs) -> list[np.ndarray]:
        """Execute the neural network.

        Arguments:
            - norm_obs: np.ndarray
                  Normalized observation to pass to the NN.
            - kwargs
                  Optional keyword arguments.

        Returns:
            - _: list[np.ndarray]
                  The normalized action distribution and optionally other
                  quantities the neural network predicted for the given input.
        """

        return self._nn.predict([np.reshape(norm_obs, (1, len(norm_obs)))])

    @abstractmethod
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

        raise NotImplementedError

    @abstractmethod
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
                  A clipped and normalized action.
        """

        raise NotImplementedError

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Transform an observation into a normalized observation.

        Arguments:
            - obs: np.ndarray
                  The observation to transform.

        Returns:
            - _: np.ndarray
                  The normalized observation.
        """

        return transform_linear(
            obs,
            self._env_config.obs_lb,
            self._env_config.obs_ub,
            self._env_config.norm_obs_lb,
            self._env_config.norm_obs_ub,
        )

    def denormalize_obs(self, norm_obs: np.ndarray) -> np.ndarray:
        """Transform a normalized observation into a regular observation.

        Arguments:
            - norm_obs: np.ndarray
                  The normalized observation to transform.

        Returns:
            - _: np.ndarray
                  The regular observation.
        """

        return transform_linear(
            norm_obs,
            self._env_config.norm_obs_lb,
            self._env_config.norm_obs_ub,
            self._env_config.obs_lb,
            self._env_config.obs_ub,
        )

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Transform an action into a normalized action.

        Arguments:
            - action: np.ndarray
                  The action to transform.

        Returns:
            - _: np.ndarray
                  The normalized action.
        """

        return transform_atanh(
            action, self._env_config.action_lb, self._env_config.action_ub
        )

    def normalize_discrete_action(self, action: np.ndarray) -> np.ndarray:
        """Normalize a discrete action.

        Arguments:
            - action: np.ndarray
                  The action to normalize.

        Return:
            - _: np.ndarray:
                  The normalized, discrete action.

        Raises:
            - A `RuntimeError` is raised if no discrete action map is available.
            - A `ValueError` is raised if the action isn't in the image of the
              discrete action map.
        """

        # Perform checks
        if self._discrete_action_map is None:
            raise RuntimeError(
                "The neural network module hasn't been given a map for discrete"
                + " actions."
            )

        for k, v in self._discrete_action_map.items():
            if v == action:
                return np.ndarray([k], dtype=np.int32)
        else:
            raise ValueError(
                f"The action '{action}' isn't in the image of the neural"
                + " network module's discrete action map."
            )

    def denormalize_action(self, norm_action: np.ndarray) -> np.ndarray:
        """Transform a normalized action into a regular action.

        Arguments:
            - norm_action: np.ndarray
                  The normalized action to transform.

        Returns:
            - _: np.ndarray
                  The regular action.
        """

        return transform_tanh(
            norm_action, self._env_config.action_lb, self._env_config.action_ub
        )

    def denormalize_discrete_action(
        self, norm_action: np.ndarray
    ) -> np.ndarray:
        """Denormalize a normalized, discrete action.

        Arguments:
            - norm_action: np.ndarray
                  The normalized action to denormalize.

        Return:
            - _: np.ndarray:
                  The denormalized action.

        Raises:
            - A `RuntimeError` is raised if no discrete action map is available.
            - A `ValueError` is raised if the discrete, normalized action isn't
              in the domain of the discrete action map.
        """

        # Perform checks
        if self._discrete_action_map is None:
            raise RuntimeError(
                "The neural network module hasn't been given a map for discrete"
                + " actions."
            )
        if norm_action.dtype != np.int32:
            logger.warning(
                "The data type of the normalized action is not `np.int32`."
                + " The data will be cast to `int`."
            )

        action = self._discrete_action_map.get(int(norm_action[0]), None)
        if action is None:
            raise ValueError(
                f"The discrete, normalized action '{norm_action}' isn't in the"
                + " domain of the neural network module's discrete action map."
            )
        return action

    @abstractmethod
    def normalize_action_dist(self, action_dist: np.ndarray) -> np.ndarray:
        """Normalize an action distribution.

        Arguments:
            - action_dist: np.ndarray
                  An action distribution.

        Returns:
            - _: np.ndarray
                  The normalized action distribution.
        """

        raise NotImplementedError

    @abstractmethod
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

        raise NotImplementedError
