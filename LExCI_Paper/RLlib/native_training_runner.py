"""Tool for running native Ray/RLlib trainings in the pendulum environment. It
replicates the way LExCI's RL Block processes observations and actions.

Depending on the software versions you have installed, you may need to edit
l. 414 of ray/rllib/utils/pre_checks/env.py in your Python environment to
    >>>     elif not isinstance(done, bool):
so that RLlib doesn't crash.

File:   native_training_runner.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2023-10-18


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

import argparse
import sys


# Parse command line arguments
arg_parser = argparse.ArgumentParser(
    description=(
        "Tool for running native Ray/RLlib trainings in the pendulum"
        + " environment. It replicates the way LExCI's RL Block processes"
        + " observations and actions."
    )
)
arg_parser.add_argument(
    "algorithm", type=str, help="RL algorithm (either 'ppo' or 'ddpg')."
)
cli_args = arg_parser.parse_args(sys.argv[1:])
if cli_args.algorithm not in ["ppo", "ddpg"]:
    raise ValueError(
        "The command line argument for the algorithm must be either 'ppo' or"
        + f" 'ddpg' but '{cli_args.algorithm}' was passed."
    )


import ray
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.ddpg as ddpg
from gym.envs.classic_control.pendulum import PendulumEnv

import copy
import numpy as np
from typing import Any


def transform_linear(
    x: np.ndarray,
    x_min: np.ndarray,
    x_max: np.ndarray,
    y_min: np.ndarray,
    y_max: np.ndarray,
) -> np.ndarray:
    """Transform data linearly between two closed intervals.

    Arguments:
        - x: np.ndarray
              Input data in [x_min, x_max].
        - x_min: np.ndarray
              Lower bound of the input.
        - x_max: np.ndarray
              Upper bound of the input.
        - y_min: np.ndarray
              Lower bound of the output.
        - y_max: np.ndarray
              Upper bound of the output.

    Returns:
        - y: np.ndarray
              Transformed data in [y_min, y_max].
    """

    y = copy.deepcopy(x)  # in [x_min, x_max]
    y = (y - x_min) / (x_max - x_min)  # in [0, 1]
    y = (y_max - y_min) * y + y_min  # in [y_min, y_max]
    return y


def transform_tanh(
    x: np.ndarray, y_min: np.ndarray, y_max: np.ndarray
) -> np.ndarray:
    """Transform from real values to a closed interval using a hyperbolic
    tangent.

    Arguments:
        - x: np.ndarray
              Input data in [-inf, inf].
        - y_min: np.ndarray
              Lower bound of the output.
        - y_max: np.ndarray
              Upper bound of the output.

    Returns:
        - y: np.ndarray
              Transformed data in [y_min, y_max].
    """

    y = copy.deepcopy(x)  # in [-inf, inf]
    y = (np.tanh(y) + 1) / 2  # in [0, 1]
    y = (y_max - y_min) * y + y_min  # in [y_min, y_max]
    return y


class NativePendulumEnv(PendulumEnv):
    """Pendulum environment where observations are min-max normalized and
    actions are mapped using a hyperbolic tangent."""

    def __init__(self, env_config: dict[str, Any]) -> None:
        """Initialize the environment.

        Arguments:
            - env_config: dict[str, Any]
                  Dictionary containing environment parameters.
        """

        # Call the initializer of the super class
        super().__init__(g=9.81)

        # Boundaries for observation normalization and action mapping
        self._obs_lb = np.array([-1, -1, -8], dtype=np.float32)
        self._obs_ub = np.array([+1, +1, +8], dtype=np.float32)
        self._action_lb = np.array([-2], dtype=np.float32)
        self._action_ub = np.array([+2], dtype=np.float32)
        self._norm_obs_lb = np.array([-1, -1, -1], dtype=np.float32)
        self._norm_obs_ub = np.array([+1, +1, +1], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        """Get the current normalized observation.

        Returns:
            - _: np.ndarray
                  The current normalized observation.
        """

        obs = super()._get_obs()
        norm_obs = transform_linear(
            obs,
            self._obs_lb,
            self._obs_ub,
            self._norm_obs_lb,
            self._norm_obs_ub,
        )
        return norm_obs

    def step(
        self, norm_action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Perform a step in the environment.

        Arguments:
            - norm_action: np.ndarray
                  The agent's normalized action.

        Returns:
            - observation: np.ndarray
                  The current normalized observation of the agent.
            - reward: float
                  The reward that is given to the agent after the previous time
                  step.
            - done: bool
                  Flag indicating whether the episode has finished.
            - info: dict[str, Any]
                  Dictionary containing auxiliary data.
        """

        action = transform_tanh(norm_action, self._action_lb, self._action_ub)
        return super().step(action)


if __name__ == "__main__":
    # Initialize Ray/RLlib
    ray.init(num_gpus=0)

    # Run the training
    if cli_args.algorithm == "ppo":
        ray.tune.run(
            "PPO",
            stop={"timesteps_total": 1000000},
            config={
                # Environment to run
                "env": NativePendulumEnv,
                # Config file parameters that are overwritten by the LExCI
                "num_workers": 0,
                "rollout_fragment_length": 512,
                "framework": "tf2",
                # Config file parameters
                "model": {
                    "fcnet_hiddens": [64, 64],
                    "fcnet_activation": "tanh",
                },
                "train_batch_size": 512,
                "sgd_minibatch_size": 64,
                "num_sgd_iter": 6,
                "gamma": 0.95,
                "lambda": 0.1,
                "clip_param": 0.3,
                "vf_clip_param": 10000,
                "lr": 0.0003,
                "kl_target": 0.01,
                "horizon": 200,
                "soft_horizon": False,
                "no_done_at_end": True,
            },
        )
    else:
        ray.tune.run(
            "DDPG",
            stop={"timesteps_total": 1000000},
            config={
                # Environment to run
                "env": NativePendulumEnv,
                # Config file parameters that are overwritten by the LExCI
                "num_workers": 0,
                "framework": "tf2",
                "use_state_preprocessor": False,
                # "_disable_execution_plan_api": True,
                # Config file parameters
                "rollout_fragment_length": 1,
                "timesteps_per_iteration": 600,
                "learning_starts": 500,
                "actor_hiddens": [64, 64],
                "actor_hidden_activation": "relu",
                "critic_hiddens": [64, 64],
                "critic_hidden_activation": "relu",
                "replay_buffer_config": {
                    "capacity": 10000,
                },
                "store_buffer_in_checkpoints": True,
                "train_batch_size": 64,
                "gamma": 0.99,
                "actor_lr": 0.001,
                "critic_lr": 0.001,
                "use_huber": True,
                "huber_threshold": 1.0,
                "l2_reg": 0.000001,
                "tau": 0.001,
                "target_network_update_freq": 0,
                "horizon": 200,
                "soft_horizon": False,
                "no_done_at_end": True,
                "training_intensity": 2500,
            },
        )

    # Terminate Ray/RLlib
    ray.shutdown()
