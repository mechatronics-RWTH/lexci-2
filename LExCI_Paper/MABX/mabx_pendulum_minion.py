"""Minion for the pendulum environment running on the MABXIII.

File:   mabx_pendulum_minion.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2023-09-25


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
from lexci2.minion.controllers.controldesk_controller import (
    ControlDeskController,
)
from lexci2.data_containers import Experience, Episode, Cycle
from lexci2.utils.transform import transform_linear, transform_tanh
from lexci2.utils.misc import get_datetime_str

import time
import logging
import os
import copy
import numpy as np
from typing import Any


# The algorithm that is used for training. This must be either 'ppo' or 'ddpg'.
# Also, remember to set the corresponding value in
# 'ReinforcementLearningBlockInit.m'. When using PPO, the S-Function
# 'PolicyNeuralNetwork' must have a two-dimensional output. For DDPG, its output
# is one-dimensional.
algorithm = "ppo"


# Logger
logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
)
logger = logging.getLogger()


# ControlDesk controller
controller = ControlDeskController()


def denormalize_obs(norm_obs: np.ndarray) -> np.ndarray:
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
        np.array([-1, -1, -1], dtype=np.float32),
        np.array([+1, +1, +1], dtype=np.float32),
        np.array([-1, -1, -8], dtype=np.float32),
        np.array([+1, +1, +8], dtype=np.float32),
    )


def denormalize_action(norm_action: np.ndarray) -> np.ndarray:
    """Transform a normalized action into a regular action.

    Arguments:
        - norm_action: np.ndarray
              The normalized action to transform.

    Returns:
        - _: np.ndarray
              The regular action.
    """

    return transform_tanh(
        norm_action,
        np.array([-2], dtype=np.float32),
        np.array([+2], dtype=np.float32),
    )


def truncate_episode(episode: Episode) -> Episode:
    """Remove excess experiences after the episode has finished.

    Arguments:
        - episode: Episode
              The episode to truncate.

    Returns:
        - _: Episode
              The truncated episode.
    """

    # Create a copy
    truncated_episode = copy.deepcopy(episode)

    # Skip the very first experience as it contains faulty data
    truncated_episode.exps = truncated_episode.exps[1:]

    # Find the true last experience
    idx = None
    for i, e in enumerate(truncated_episode):
        if e.done:
            idx = i
            break

    # Remove excess experiences
    if idx is not None:
        while len(truncated_episode.exps) > idx + 1:
            del truncated_episode.exps[-1]

    return truncated_episode


def postprocess_episode(episode: Episode) -> Episode:
    """Add denormalized quantities as auxiliary data to an episode.

    Arguments:
        - episode: Episode
              The episode to post-process.

    Returns:
        - _: Episode
              The post-processed episode.
    """

    eps = copy.deepcopy(episode)
    for e in eps:
        denorm_obs = denormalize_obs(e.obs)
        denorm_action = denormalize_action(e.action)
        denorm_new_obs = denormalize_obs(e.new_obs)

        e.aux_data = {
            "denorm_obs": denorm_obs.tolist(),
            "denorm_action": denorm_action.tolist(),
            "denorm_new_obs": denorm_new_obs.tolist(),
        }

        e.done = False

    return eps


def calc_stdev(
    initial_stdev: float, stdev_decay_factor: float, cycle_no: int
) -> float:
    """Calculate the standard deviation of the action distribution as a function
    of the current LExCI cycle.

    Arguments:
        - initial_stdev: float
              Initial standard deviation, i.e. at cycle 0.
        - stdev_decay_factor: float
              Factor in (0, 1) that governs how quickly the standard deviation
              converges to zero.
        - cycle_no: int
              Current LExCI cycle number.

    Returns:
        - _: float
              The standard deviation of the action distribution.
    """

    return initial_stdev * stdev_decay_factor**cycle_no


def prepare_env(
    nn_bytes: bytes,
    cycle_no: int,
    minion_params: dict[str, Any],
    is_training_episode: bool,
) -> None:
    """Prepare the pendulum environment on the MABXIII for a training or a
    validation run.

    Arguments:
        - nn_bytes: bytes
              Bytes of the TensorFlow Lite model for the agent's behavior.
        - cycle_no: int
              Current LExCI cycle number.
        - minion_params: dict[str, Any]
              Miscellaneous parameters.
        - is_training_episode: bool
              Whether to prepare the environment for a training run (`True`) or
              a validation run (`False`).
    """

    # Reset the environment
    controller.write_var(r"Model Root/Episode_Scheduler/bEpStartReq/Value", 0)
    controller.write_var(r"Model Root/Episode_Scheduler/bEpStopReq/Value", 1)
    time.sleep(1)
    controller.write_var(r"Model Root/Episode_Scheduler/bEpStopReq/Value", 0)

    # Overwrite the RL Block's adjustable parameters
    controller.write_var(
        r"Model Root/RL_Block/RL_Agent/norm_observation_size/Value",
        3,
    )
    if algorithm == "ddpg":
        controller.write_var(
            r"Model Root/RL_Block/RL_Agent/norm_action_dist_size/Value",
            1,
        )
    elif algorithm == "ppo":
        controller.write_var(
            r"Model Root/RL_Block/RL_Agent/norm_action_dist_size/Value",
            2,
        )
    controller.write_var(
        r"Model Root/RL_Block/RL_Agent/static_nn_memory/Value",
        list(nn_bytes),
    )
    controller.write_var(
        r"Model Root/RL_Block/RL_Agent/tensor_arena_size/Value",
        100000,
    )
    controller.write_var(
        r"Model Root/RL_Block/RL_Agent/lower_observation_bounds/Value",
        [-1, -1, -8],
    )
    controller.write_var(
        r"Model Root/RL_Block/RL_Agent/upper_observation_bounds/Value",
        [+1, +1, +8],
    )
    controller.write_var(
        r"Model Root/RL_Block/RL_Agent/lower_action_bounds/Value",
        [-2],
    )
    controller.write_var(
        r"Model Root/RL_Block/RL_Agent/upper_action_bounds/Value",
        [+2],
    )
    if is_training_episode:
        controller.write_var(
            r"Model Root/RL_Block/RL_Agent/b_training_active/Value",
            1,
        )
    else:
        controller.write_var(
            r"Model Root/RL_Block/RL_Agent/b_training_active/Value",
            0,
        )

    # Set the standard deviation for DDPG's sampling system
    if algorithm == "ddpg":
        initial_stdev = minion_params.get("INITIAL_STDEV", 0.5)
        stdev_decay_factor = minion_params.get("STDEV_DECAY_FACTOR", 0.925)
        controller.write_var(
            r"Model Root/RL_Block/RL_Agent/ddpg_standard_deviation/Value",
            calc_stdev(initial_stdev, stdev_decay_factor, cycle_no),
        )

    # Set the seeds of the RNGs
    controller.write_var(
        r"Model Root/RL_Block/RL_Agent/Sampling/Continuous Action/DDPG_Sampler/Random\nNumber/Seed",
        int(time.time()),
    )
    controller.write_var(
        r"Model Root/RL_Block/RL_Agent/Sampling/Continuous Action/PPO_Sampler/Random\nNumber/Seed",
        int(time.time()),
    )

    # Set the initial values
    if is_training_episode:
        controller.write_var(
            r"Model Root/theta_init/Value",
            np.random.uniform(-np.pi, +np.pi),
        )
        controller.write_var(
            r"Model Root/thetadot_init/Value",
            np.random.uniform(-1, +1),
        )
    else:
        controller.write_var(r"Model Root/theta_init/Value", np.pi)
        controller.write_var(r"Model Root/thetadot_init/Value", 0.0)


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

    num_collected_exps = 0
    num_episodes = 1
    cycle = Cycle()

    print(f"========== Training Cycle {cycle_no} ==========")
    while num_collected_exps < num_experiences:
        print(f"Starting episode {num_episodes}.")

        # Prepare the environment
        prepare_env(model_bytes, cycle_no, minion_params, True)

        # Start the simulation and wait for it to end
        output_file_name = os.path.join(
            r"D:\LExCI_Paper\Experiments\ControlDesk_Recorder_CSVs",
            get_datetime_str() + ".csv",
        )
        controller.start_triggered_recording(output_file_name)
        time.sleep(1)
        controller.write_var(
            r"Model Root/Episode_Scheduler/bEpStartReq/Value", 1
        )
        time.sleep(1)
        controller.write_var(
            r"Model Root/Episode_Scheduler/bEpStartReq/Value", 0
        )
        while True:
            episode_done = controller.read_var(
                r"Model Root/RL_Block/RL_Agent/Experience Buffer/b_episode_finished_out/Out1"
            )[0]
            if episode_done == 1:
                controller.stop_triggered_recording()
                break
            time.sleep(5)

        # Retrieve the data
        while not controller.is_csv_ready(output_file_name):
            time.sleep(5)
        episode = controller.extract_csv_data(output_file_name, True)[0]
        episode = truncate_episode(episode)
        episode = postprocess_episode(episode)
        cycle.add_episode(episode)
        num_collected_exps += len(episode)
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

    print(f"========== Validation Cycle {cycle_no} ==========")

    # Prepare the environment
    prepare_env(model_bytes, cycle_no, minion_params, False)

    # Start the simulation and wait for it to end
    output_file_name = os.path.join(
        r"D:\LExCI_Paper\Experiments\ControlDesk_Recorder_CSVs",
        get_datetime_str() + ".csv",
    )
    controller.start_triggered_recording(output_file_name)
    time.sleep(1)
    controller.write_var(r"Model Root/Episode_Scheduler/bEpStartReq/Value", 1)
    time.sleep(1)
    controller.write_var(r"Model Root/Episode_Scheduler/bEpStartReq/Value", 0)
    while True:
        episode_done = controller.read_var(
            r"Model Root/RL_Block/RL_Agent/Experience Buffer/b_episode_finished_out/Out1"
        )[0]
        if episode_done == 1:
            controller.stop_triggered_recording()
            break
        time.sleep(5)

    # Retrieve the data
    while not controller.is_csv_ready(output_file_name):
        time.sleep(5)
    episode = controller.extract_csv_data(output_file_name, True)[0]
    episode = truncate_episode(episode)
    episode = postprocess_episode(episode)
    cycle = Cycle()
    cycle.add_episode(episode)
    return cycle


if __name__ == "__main__":
    minion = Minion(
        "192.168.56.101", 5555, generate_training_data, generate_validation_data
    )
    controller.start_controldesk(
        r"D:\LExCI_Paper\Experiments\ControlDesk\MABX_Pendulum_Env\MABX_Pendulum_Env.CDP",
        "Pendulum_Env",
    )
    minion.mainloop()
    controller.stop_controldesk()
