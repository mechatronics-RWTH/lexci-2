"""A universal LExCI 2 Master that uses a DDPG agent and is configurable via a
JSON file.

File:   lexci2/universal_masters/universal_ddpg_master/universal_ddpg_master.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2022-08-24


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
from lexci2.ray_controller import start_ray, stop_ray
from lexci2.lexci_env import LexciEnvConfig
from lexci2.agents.ddpg_agent import DdpgAgent
from lexci2.master.master import Master

import sys
import os
import argparse
import json
import logging
import numpy as np


# Logger
logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
)
logger = logging.getLogger()


def main() -> None:
    """Main function of the universal Master."""

    # Command line arguments
    arg_parser = argparse.ArgumentParser(
        description="Universal DDPG LExCI 2 Master."
    )
    arg_parser.add_argument(
        "config_json_file", type=str, help="JSON-formatted configuration file."
    )
    cli_args = arg_parser.parse_args(sys.argv[1:])

    # Load the config JSON
    with open(cli_args.config_json_file, "r") as f:
        config = json.load(f)

    # LExCI environment configuration
    obs_size = config["master_config"]["obs_size"]
    action_size = config["master_config"]["action_size"]
    obs_lb = -1 * np.ones(obs_size, dtype=np.float32)
    obs_ub = +1 * np.ones(obs_size, dtype=np.float32)
    action_lb = -1 * np.ones(action_size, dtype=np.float32)
    action_ub = +1 * np.ones(action_size, dtype=np.float32)
    norm_obs_lb = -1 * np.ones(obs_size, dtype=np.float32)
    norm_obs_ub = +1 * np.ones(obs_size, dtype=np.float32)
    norm_action_lb = -np.inf * np.ones(action_size, dtype=np.float32)
    norm_action_ub = +np.inf * np.ones(action_size, dtype=np.float32)
    env_config = LexciEnvConfig(
        obs_size,
        action_size,
        "continuous",
        obs_lb,
        obs_ub,
        action_lb,
        action_ub,
        norm_obs_lb,
        norm_obs_ub,
        norm_action_lb,
        norm_action_ub,
    )

    # DDPG agent and its configuration
    ddpg_config = DdpgAgent.get_default_trainer_config()
    ddpg_config.update(config["ddpg_config"])
    agent = DdpgAgent("agent0", env_config, ddpg_config)

    # Create the Master
    start_ray()
    master = Master(
        agent,
        addr=config["master_config"]["addr"],
        port=config["master_config"]["port"],
        num_experiences_per_cycle=config["master_config"][
            "num_experiences_per_cycle"
        ],
        mailbox_buffer_size=config["master_config"]["mailbox_buffer_size"],
        min_num_minions=config["master_config"]["min_num_minions"],
        max_num_minions=config["master_config"]["max_num_minions"],
        minion_job_timeout=config["master_config"]["minion_job_timeout"],
        minion_params=config["master_config"]["minion_params"],
        nn_format=config["master_config"]["nn_format"],
        nn_size=config["master_config"]["nn_size"],
        output_dir=config["master_config"]["output_dir"],
        b_save_training_data=config["master_config"]["b_save_training_data"],
        b_save_sample_batches=config["master_config"]["b_save_sample_batches"],
        validation_interval=config["master_config"]["validation_interval"],
        num_replay_trainings=config["master_config"]["num_replay_trainings"],
        perc_replay_trainings=config["master_config"]["perc_replay_trainings"],
        num_exp_before_replay_training=config["master_config"][
            "num_exp_before_replay_training"
        ],
        offline_data_import_folder=config["master_config"][
            "offline_data_import_folder"
        ],
        b_offline_training_only=(
            config["master_config"]["b_offline_training_only"]
        ),
    )
    if config["master_config"]["checkpoint_file"] != "":
        master.restore_checkpoint(config["master_config"]["checkpoint_file"])
    elif config["master_config"]["model_h5_folder"] != "":
        master.import_models_h5(config["master_config"]["model_h5_folder"])

    # Create a copy of the config file in the log directory
    with open(cli_args.config_json_file, "r") as f:
        s_config = f.read()
    copied_config_file_name = os.path.join(
        master.get_output_dir(), os.path.basename(cli_args.config_json_file)
    )
    with open(copied_config_file_name, "w") as f:
        f.write(s_config)

    # If a documentation string has been passed in the config, write its content
    # into a text file in the output directory of the training
    if config["master_config"]["doc"] != "":
        doc_file_name = os.path.join(
            master.get_output_dir(), "Documentation.txt"
        )
        with open(doc_file_name, "w") as f:
            f.write(config["master_config"]["doc"])

    # Run the Master
    master.start()
    stop_ray()


if __name__ == "__main__":
    main()
