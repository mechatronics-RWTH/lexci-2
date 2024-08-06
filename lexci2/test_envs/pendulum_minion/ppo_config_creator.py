"""Script for creating the universal PPO master's configuration file.

File:   lexci2/test_envs/pendulum_minion/ppo_config_creator.py
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

import sys
import argparse
import copy
import json

from typing import Any


# Parse command line arguments
arg_parser = argparse.ArgumentParser(
    description=(
        "Export the Universal PPO"
        " LExCI 2 Master's configuration to a JSON file."
    )
)
arg_parser.add_argument("output_file", type=str, help="Output file to write.")
cli_args = arg_parser.parse_args(sys.argv[1:])


# Master configuration dictrionary
master_config = {}
# =========================== MAKE ADJUSTMENTS HERE ============================#
master_config["obs_size"] = 3
master_config["action_size"] = 1
master_config["action_type"] = "continuous"
master_config["addr"] = "0.0.0.0"
master_config["port"] = 5555
master_config["mailbox_buffer_size"] = 1 * 1024**3
master_config["min_num_minions"] = 1
master_config["max_num_minions"] = 2
master_config["minion_job_timeout"] = 3600.0
master_config["minion_params"] = {
    "render": False,
}
master_config["nn_format"] = "tflite"
master_config["nn_size"] = 64 * 1024**1
master_config["output_dir"] = "~/lexci_results"
master_config["b_save_training_data"] = False
master_config["b_save_sample_batches"] = False
master_config["validation_interval"] = 5
master_config["checkpoint_file"] = ""
master_config["model_h5_folder"] = ""
# If the documentation string isn't empty, the universal Master will create a
# text file called 'Documentation.txt' in the training's log directory and write
# the content of the string into said file.
master_config["doc"] = ""
# ==============================================================================#


# PPO configuration dictionary
import ray.rllib.agents.ppo as ppo

ppo_config = copy.deepcopy(ppo.DEFAULT_CONFIG)
# =========================== MAKE ADJUSTMENTS HERE ============================#
ppo_config["model"]["fcnet_hiddens"] = [64, 64]
ppo_config["model"]["fcnet_activation"] = "tanh"
ppo_config["train_batch_size"] = 512  # = Number of experiences per cycle
ppo_config["sgd_minibatch_size"] = 64
ppo_config["num_sgd_iter"] = 6
ppo_config["gamma"] = 0.95
ppo_config["lambda"] = 0.1
ppo_config["clip_param"] = 0.3
ppo_config["vf_clip_param"] = 10000
ppo_config["lr"] = 0.0003
ppo_config["kl_target"] = 0.01
# ppo_config["lr_schedule"] = [[0, 0.0001], [1e9, 1e-5]]
# ==============================================================================#
# Remove keys that aren't JSON-serializable
keys_to_remove = []
for k, v in ppo_config.items():
    if v is not None and type(v) not in [dict, list, str, int, float, bool]:
        print(
            f"Removing key '{k}' with value '{v}' from the PPO configuration as"
            " it isn't JSON-serializable."
        )
        keys_to_remove.append(k)
for k in keys_to_remove:
    del ppo_config[k]


# Write the JSON file
config = {"master_config": master_config, "ppo_config": ppo_config}
with open(cli_args.output_file, "w") as f:
    json.dump(config, f, indent=2)
