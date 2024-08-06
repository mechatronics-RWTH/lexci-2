"""Script for creating the universal DDPG master's configuration file.

File:   lexci2/tests/pendulum_minion/ddpg_config_creator.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2023-07-27


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

import sys
import argparse
import copy
import json

from typing import Any


# Parse command line arguments
arg_parser = argparse.ArgumentParser(
    description=(
        "Export the Universal DDPG LExCI 2 Master's configuration to a JSON"
        + " file."
    )
)
arg_parser.add_argument("output_file", type=str, help="Output file to write.")
cli_args = arg_parser.parse_args(sys.argv[1:])


# Master configuration dictrionary
master_config = {}
# =========================== MAKE ADJUSTMENTS HERE ============================#
master_config["obs_size"] = 3
master_config["action_size"] = 1
master_config["addr"] = "0.0.0.0"
master_config["port"] = 5555
master_config["num_experiences_per_cycle"] = 600
master_config["mailbox_buffer_size"] = 1 * 1024**3
master_config["min_num_minions"] = 1
master_config["max_num_minions"] = 2
master_config["minion_job_timeout"] = 3600.0
master_config["minion_params"] = {
    "render": False,
    "INITIAL_STDEV": 0.5,
    "STDEV_DECAY_FACTOR": 0.925,
}
master_config["nn_format"] = "tflite"
master_config["nn_size"] = 64 * 1024**1
master_config["output_dir"] = "~/lexci_results"
master_config["b_save_training_data"] = False
master_config["b_save_sample_batches"] = False
master_config["validation_interval"] = 5
master_config["num_replay_trainings"] = 0
master_config["perc_replay_trainings"] = 0.25
master_config["num_exp_before_replay_training"] = (
    4 * master_config["num_experiences_per_cycle"]
)
master_config["offline_data_import_folder"] = ""
master_config["b_offline_training_only"] = False
master_config["checkpoint_file"] = ""
master_config["model_h5_folder"] = ""
# If the documentation string isn't empty, the universal Master will create a
# text file called 'Documentation.txt' in the training's log directory and write
# the content of the string into said file.
master_config["doc"] = ""
# ==============================================================================#


# PPO configuration dictionary
import ray.rllib.agents.ddpg as ddpg

ddpg_config = copy.deepcopy(ddpg.DEFAULT_CONFIG)
# =========================== MAKE ADJUSTMENTS HERE ============================#
ddpg_config["actor_hiddens"] = [64, 64]
ddpg_config["actor_hidden_activation"] = "relu"
ddpg_config["critic_hiddens"] = [64, 64]
ddpg_config["critic_hidden_activation"] = "relu"
ddpg_config["replay_buffer_config"]["capacity"] = 10000
ddpg_config["store_buffer_in_checkpoints"] = True
ddpg_config["train_batch_size"] = 64
# ddpg_config["no_done_at_end"] = True
ddpg_config["gamma"] = 0.99
ddpg_config["actor_lr"] = 0.001
ddpg_config["critic_lr"] = 0.001
ddpg_config["use_huber"] = True
ddpg_config["huber_threshold"] = 1.0
ddpg_config["l2_reg"] = 1e-6
# Update target networks using `tau*policy + (1 - tau)*target_policy`
ddpg_config["tau"] = 0.001
ddpg_config["target_network_update_freq"] = 0
# ==============================================================================#
# Remove keys that aren't JSON-serializable
keys_to_remove = []
for k, v in ddpg_config.items():
    if v is not None and type(v) not in [dict, list, str, int, float, bool]:
        print(
            f"Removing key '{k}' with value '{v}' from the DDPG configuration as"
            + " it isn't JSON-serializable."
        )
        keys_to_remove.append(k)
for k in keys_to_remove:
    del ddpg_config[k]


# Write the JSON file
config = {"master_config": master_config, "ddpg_config": ddpg_config}
with open(cli_args.output_file, "w") as f:
    json.dump(config, f, indent=2)
