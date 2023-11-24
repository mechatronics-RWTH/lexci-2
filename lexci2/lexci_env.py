"""Custom version of an external Ray environment.

File:   lexci2/lexci_env.py
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


import ray
from ray.rllib.env.external_env import ExternalEnv
import gym

import copy
import numpy as np
from typing import Optional


class LexciEnvConfig:
    """Configuration container for a LExCI environment."""

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        action_type: str,
        obs_lb: np.ndarray,
        obs_ub: np.ndarray,
        action_lb: np.ndarray,
        action_ub: np.ndarray,
        norm_obs_lb: Optional[np.ndarray] = None,
        norm_obs_ub: Optional[np.ndarray] = None,
        norm_action_lb: Optional[np.ndarray] = None,
        norm_action_ub: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize the environment config.

        Normalized parameters refer to the quantities as they are passed to or
        received from the neural network while the regular bounds pertain to the
        actual problem.
        To illustrate the difference, assume that the task is to control the
        longitudinal acceleration of a car, i.e. to learn a cruise control
        function. Let the observation space consist of the current velocity, the
        speed limit, and the distance to the preceding vehicle as well as its
        speed. Further, let the action space contain the desired acceleration.
        In this scenario, the regular bounds (i.e. the arguments without the
        'norm_'-prefix) are set to the physical/plausible/legal limits of the
        problem, for instance to the following values:

          >>> # [v_ego=0 km/h, v_limit=0 km/h, d_prec=0 m, v_prec=0 km/h]
          >>> obs_lb = np.array([0, 0, 0, 0], dtype=np.float32)
          >>> # [v_ego=200 km/h, v_limit=130 km/h, d_prec=500 m,
          >>> # v_prec=200 km/h]
          >>> obs_ub = np.array([200, 130, 500, 200], dtype=np.float32)
          >>> # [a_ego=-10 m/s^2]
          >>> action_lb = np.array([-10], dtype=np.float32)
          >>> # [a_ego=5 m/s^2]
          >>> action_ub = np.array([5], dtype=np.float32)

        Although training with these settings alone can yield good results, one
        may want the algorithm (and by extension the NN) to work with different
        spaces under the hood. For example, it is a common practice in ML/RL to
        min-max normalize the observation space to the range [-1, 1] to ensure
        that the individual observations contribute equally. Likewise, one may
        want to limit the NN's action space lest actions become too extreme.
        This is what the normalized bounds are for and they are set like this:

          >>> # Min-max normalize the NN's observation space
          >>> norm_obs_lb = np.array([-1, -1, -1, -1], dtype=np.float32)
          >>> norm_obs_ub = np.array([1, 1, 1, 1], dtype=np.float32)
          >>> # Let the normalized actions be real-valued
          >>> norm_action_lb = -np.inf*np.ones((1,), dtype=np.float32)
          >>> norm_action_ub = +np.inf*np.ones((1,), dtype=np.float32)

        Unfortunately, when LExCI and RLlib are paired, the number of options
        one has for the normalized action space is somewhat limited.  While it
        is possible, for  example,  to manipulate the policy model of an agent
        so that its output layer is tanh-activated and its actions therefore in
        [-1, 1], this can cause tremendous problems during the post-processing
        step in the Master when the train batches are assembled. The advised
        approach is to leave the normalized action space at [-inf, +inf].

        To reiterate, the algorithm operates on the normalized spaces. This
        especially means that all data stored in experiences must be normalized.
        The onus is on the user to transform quantities back and forth between
        the various spaces when executing the agent and applying its output. For
        that purpose, there are helper functions in 'lexci2/utils/transform.py'.
        Assume that the cruise control function above is trained in a simulated
        environment. Transformations can be performed as shown below:

          >>> from lexci2.utils.transform import (
          >>>     transform_linear, transform_tanh
          >>> )
          >>> from lexci2.data_containers import Experience
          >>>
          >>> sim = Simulation()  # Some handle to the simulation
          >>> nn = NeuralNetwork()  # Some handle to the agent's neural network
          >>>
          >>> # For example, obs = [23 km/h, 50 km/h, 72 m, 48 km/h]
          >>> obs = sim.get_obs()
          >>> # norm_obs = [-0.77, -0.23, -0.71, -0.52] for the values above
          >>> norm_obs = transform_linear(obs, obs_lb, obs_ub, norm_obs_lb,
          >>>                             norm_obs_ub)
          >>>
          >>> # Feed the normalized observation to the agent in order to get the
          >>> # acceleration. Here, assume norm_action = [0.8].
          >>> norm_action = nn.execute(norm_obs)
          >>> # De-normalize the agent's output. It is action = [2.48 m/s^2] for
          >>> # this example.
          >>> action = transform_tanh(norm_action, action_lb, action_ub)
          >>>
          >>> # Apply the agent's action in the simulation and gather important
          >>> # data
          >>> sim.perform_action(action)
          >>> new_obs = sim.get_obs()
          >>> reward = sim.get_reward()
          >>> done = sim.is_done()
          >>> # Get additional quantities if needed
          >>>
          >>> # Save the experience
          >>> norm_new_obs = transform_linear(
          >>>     new_obs, obs_lb, obs_ub, norm_obs_lb, norm_obs_ub
          >>> )
          >>> experience = Experience(
          >>>     norm_obs, norm_action, norm_new_obs, reward, done
          >>> )

        Arguments:
            - obs_size: int
                  Size of the observation space.
            - action_size: int
                  Size of the action space.
            - action_type: str
                  Type of the action space. This must be either 'discrete' or
                  'continuous'.
            - obs_lb: np.ndarray
                  Lower bounds of the values in the observation space.
            - obs_ub: np.ndarray
                  Upper bounds of the values in the observation space.
            - action_lb: np.ndarray
                  Lower bounds of the values in the continuous action space.
            - action_ub: np.ndarray
                  Upper bounds of the values in the continuous action space.
            - norm_obs_lb: Optional[np.ndarray] (Default: None)
                  Lower bounds of the normalized observation space. If not
                  specified, this is set to all -1's.
            - norm_obs_ub: Optional[np.ndarray] (Default: None)
                  Upper bounds of the normalized observation space. If not
                  specified, this is set to all +1's.
            - norm_action_lb: Optional[np.ndarray] (Default: None)
                  Lower bounds of the normalized action space. If not specified,
                  this is set to all -inf.
            - norm_action_ub: Optional[np.ndarray] (Default: None)
                  Upper bounds of the normalized action space. If not specified,
                  this is set to all +inf.

        Raises:
            - ValueError
        """

        # Observation space
        self.obs_size = obs_size
        self.obs_lb = copy.deepcopy(obs_lb)
        self.obs_ub = copy.deepcopy(obs_ub)

        # Action space
        if action_type not in ["discrete", "continuous"]:
            raise ValueError(
                "`action_type` must be either 'discrete' or" + " 'continuous'"
            )
        self.action_type = action_type
        self.action_size = action_size
        self.action_lb = copy.deepcopy(action_lb)
        self.action_ub = copy.deepcopy(action_ub)

        # Normalized observation space
        self.norm_obs_lb = copy.deepcopy(norm_obs_lb)
        self.norm_obs_ub = copy.deepcopy(norm_obs_ub)
        if self.norm_obs_lb is None:
            self.norm_obs_lb = -1 * np.ones((self.obs_size,), dtype=np.float32)
        if self.norm_obs_ub is None:
            self.norm_obs_ub = +1 * np.ones((self.obs_size,), dtype=np.float32)

        # Normalized action space
        self.norm_action_lb = copy.deepcopy(norm_action_lb)
        self.norm_action_ub = copy.deepcopy(norm_action_ub)
        if self.norm_action_lb is None:
            self.norm_action_lb = -np.inf * np.ones(
                (self.action_size,), dtype=np.float32
            )
        if self.norm_action_ub is None:
            self.norm_action_ub = +np.inf * np.ones(
                (self.action_size,), dtype=np.float32
            )


class LexciEnv(ExternalEnv):
    """LExCI's custom external environment.

    It extends RLlib's ExternalEnv class such that it cannot be directly used
    for sampling. Instead, it's only meant to tell the trainer how the
    observation and action space are defined. Their boundaries differ on purpose
    from what is specified in the `LexciEnvConfig`. As far as RLlib is
    concerned, both the observation and action space are treated as real-valued
    in order to circumnavigate problems that occur when proceeding otherwise. It
    is LExCI's responsibility to ensure that observations and actions conform to
    the user's wishes.
    """

    def __init__(self, env_config: LexciEnvConfig) -> None:
        """Initialize the environment.

        Arguments:
            - env_config: LexciEnvConfig
                  Configuration container for the environment.
        """

        # Observation space
        obs_lb = -np.inf * np.ones((env_config.obs_size,), dtype=np.float32)
        obs_ub = +np.inf * np.ones((env_config.obs_size,), dtype=np.float32)
        obs_space = gym.spaces.Box(
            shape=(env_config.obs_size,),
            low=obs_lb,
            high=obs_ub,
            dtype=np.float32,
        )

        # Action space
        if env_config.action_type == "discrete":
            action_space = gym.spaces.Discrete(env_config.action_size)
        else:
            action_lb = -np.inf * np.ones(
                (env_config.action_size,), dtype=np.float32
            )
            action_ub = +np.inf * np.ones(
                (env_config.action_size,), dtype=np.float32
            )
            action_space = gym.spaces.Box(
                shape=(env_config.action_size,),
                low=action_lb,
                high=action_ub,
                dtype=np.float32,
            )

        super().__init__(action_space, obs_space)

    def run(self) -> None:
        """Override the method so that it does nothing."""

        pass
