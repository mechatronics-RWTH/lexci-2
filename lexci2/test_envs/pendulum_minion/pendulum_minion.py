"""Minion for the Pendulum environment.

File:   lexci2/test_envs/pendulum_minion/pendulum_minion.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2023-07-24


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

import lexci2
from lexci2.minion.minion import Minion
from lexci2.neural_network_modules.continuous_ppo_neural_network_module import (
    ContinuousPpoNeuralNetworkModule,
)
from lexci2.neural_network_modules.ddpg_neural_network_module import (
    DdpgNeuralNetworkModule,
)
from lexci2.lexci_env import LexciEnvConfig
from lexci2.data_containers import Experience, Episode, Cycle
from lexci2.utils.transform import transform_tanh, transform_atanh
from gym.envs.classic_control import PendulumEnv

import logging
import argparse
import sys
import numpy as np
from typing import Any


# Logger
logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
)
logger = logging.getLogger()


# Parse command line arguments
arg_parser = argparse.ArgumentParser(
    description="Minion for the Pendulum environment."
)
arg_parser.add_argument(
    "algorithm", type=str, help="RL algorithm (either 'ppo', 'ddpg', or 'td3')."
)
cli_args = arg_parser.parse_args(sys.argv[1:])


# LExCI environment
env_config = LexciEnvConfig(
    3,
    1,
    "continuous",
    obs_lb=np.array([-1, -1, -8], dtype=np.float32),
    obs_ub=np.array([+1, +1, +8], dtype=np.float32),
    action_lb=np.array([-2], dtype=np.float32),
    action_ub=np.array([+2], dtype=np.float32),
    norm_obs_lb=np.array([-1, -1, -1], dtype=np.float32),
    norm_obs_ub=np.array([1, 1, 1], dtype=np.float32),
    norm_action_lb=np.array([-np.inf], dtype=np.float32),
    norm_action_ub=np.array([+np.inf], dtype=np.float32),
)


# Test environment
pendulum_env = PendulumEnv(g=9.81)


def print_transition(
    obs: np.ndarray,
    action: np.ndarray,
    reward: float,
    action_dist: np.ndarray,
) -> None:
    """Print a system transition in a human-legible form.

    Arguments:
        - obs: np.ndarray
              The denormalized observation of the agent.
        - action: np.ndarray:
              The denormalized action the agent chose.
        - reward: float
              The reward for the transition.
        - action_dist: np.ndarray
              The denormalized action distribution for the transition.
    """

    s = ""

    # Observations
    s += "["
    for i in range(len(obs)):
        s += f"{obs[i]:+05.2f}"
        if i < len(obs) - 1:
            s += ", "
    s += "]"

    # Actions
    s += "\t["
    for i in range(len(action)):
        s += f"{action[i]:+05.2f}"
        if i < len(action) - 1:
            s += ", "
    s += "]"

    # Reward
    s += f"\t{reward:+06.2f}"

    # Action distribution
    if cli_args.algorithm == "ppo":
        means, stdevs = np.split(action_dist, 2)
        s += "\t["
        for i in range(len(means)):
            s += f"({means[i]:+05.2f}, {stdevs[i]:+05.2f})"
            if i < len(means) - 1:
                s += ", "
        s += "]"
    elif cli_args.algorithm in ["ddpg", "td3"]:
        s += "\t["
        for i in range(len(action_dist)):
            s += f"{action_dist[i]:+06.2f}"
            if i < len(action_dist) - 1:
                s += ", "
        s += "]"

    print(s)


def generate_training_data(
    model_bytes: bytes,
    cycle_no: int,
    num_experiences: int,
    minion_params: dict[str, Any],
) -> Cycle:
    """Generate training data.

    Arguments:
      - model_bytes: bytes
          Bytes of the TensorFlow Lite model for the agent's behavior.
      - cycle_no: int
          Current LExCI cycle number.
      - num_experiences: int
          Number of experiences to generate.
      - minion_params: dict[str, Any]
          Miscellaneous parameters.

    Returns:
      - _: Cycle
          `Cycle` object containing the generated data.
    """

    if cli_args.algorithm == "ppo":
        nn_module = ContinuousPpoNeuralNetworkModule(
            env_config, model_bytes, "tflite"
        )
    elif cli_args.algorithm in ["ddpg", "td3"]:
        nn_module = DdpgNeuralNetworkModule(env_config, model_bytes, "tflite")
    else:
        raise RuntimeError(f"Unknown algorithm '{cli_args.algorithm}'.")
    num_collected_exps = 0
    num_episodes = 1
    cycle = Cycle()

    print(f"========== Training Cycle {cycle_no} ==========")
    while num_collected_exps < num_experiences:
        print(f"----- Episode {num_episodes} -----")
        episode = Episode("agent0")
        pendulum_env.reset()

        num_steps = 0
        while num_steps < 200:
            obs = pendulum_env._get_obs()
            norm_obs = nn_module.normalize_obs(obs)
            norm_action = nn_module.get_norm_action(norm_obs)
            norm_action_dist = nn_module.get_norm_action_dist(norm_obs)
            action = nn_module.denormalize_action(norm_action)
            action_dist = nn_module.denormalize_action_dist(norm_action_dist)

            _, reward, done, _ = pendulum_env.step(action)

            new_obs = pendulum_env._get_obs()
            norm_new_obs = nn_module.normalize_obs(new_obs)
            episode.append_experience(
                Experience(
                    norm_obs,
                    norm_action,
                    norm_new_obs,
                    reward,
                    done,
                    aux_data={
                        "norm_action_dist": norm_action_dist.tolist(),
                        "denorm_obs": obs.tolist(),
                        "denorm_action": action.tolist(),
                        "denorm_new_obs": new_obs.tolist(),
                        "denorm_action_dist": action_dist.tolist(),
                    },
                )
            )
            num_collected_exps += 1
            num_steps += 1
            print_transition(obs, action, reward, action_dist)

        print("\n")
        cycle.add_episode(episode)
        num_episodes += 1

    return cycle


def generate_validation_data(
    model_bytes: bytes, cycle_no: int, minion_params: dict[str, Any]
) -> Cycle:
    """Generate training data.

    Arguments:
      - model_bytes: bytes
          Bytes of the TensorFlow Lite model for the agent's behavior.
      - cycle_no: int
          Current LExCI cycle number.
      - minion_params: dict[str, Any]
          Miscellaneous parameters.

    Returns:
      - _: Cycle
          `Cycle` object containing the generated data.
    """

    if cli_args.algorithm == "ppo":
        nn_module = ContinuousPpoNeuralNetworkModule(
            env_config, model_bytes, "tflite"
        )
    elif cli_args.algorithm in ["ddpg", "td3"]:
        nn_module = DdpgNeuralNetworkModule(env_config, model_bytes, "tflite")
    else:
        raise RuntimeError(f"Unknown algorithm '{cli_args.algorithm}'.")

    cycle = Cycle()

    print(f"========== Validation Cycle {cycle_no} ==========")
    episode = Episode("agent0")
    pendulum_env.reset()
    # For validations, the pendulum is always at the 6 o'clock position with no
    # angular velocity
    pendulum_env.state = np.array([np.pi, 0], dtype=np.float32)

    num_steps = 0
    while num_steps < 200:
        obs = pendulum_env._get_obs()
        norm_obs = nn_module.normalize_obs(obs)
        norm_action = nn_module.get_norm_action(norm_obs, False)
        norm_action_dist = nn_module.get_norm_action_dist(norm_obs)
        action = nn_module.denormalize_action(norm_action)
        action_dist = nn_module.denormalize_action_dist(norm_action_dist)

        _, reward, done, _ = pendulum_env.step(action)

        new_obs = pendulum_env._get_obs()
        norm_new_obs = nn_module.normalize_obs(new_obs)
        episode.append_experience(
            Experience(
                norm_obs,
                norm_action,
                norm_new_obs,
                reward,
                done,
                aux_data={
                    "norm_action_dist": norm_action_dist.tolist(),
                    "denorm_obs": obs.tolist(),
                    "denorm_action": action.tolist(),
                    "denorm_new_obs": new_obs.tolist(),
                    "denorm_action_dist": action_dist.tolist(),
                },
            )
        )

        num_steps += 1
        print_transition(obs, action, reward, action_dist)

    cycle.add_episode(episode)
    return cycle


if __name__ == "__main__":
    minion = Minion(
        "127.0.0.1", 5555, generate_training_data, generate_validation_data
    )
    minion.mainloop()
