"""Script for creating the universal PPO master's configuration file.

File:   lexci2/universal_masters/universal_ppo_master/ppo_config_creator.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2022-08-12


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

import copy
import logging

from lexci2.universal_masters.config_creator import ConfigCreator


# Create the logger
logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
)
logger = logging.getLogger()


# Master configuration dictrionary
master_config = {}
# =========================== MAKE ADJUSTMENTS HERE ===========================#
master_config["obs_size"] = 10
master_config["action_size"] = 1
master_config["action_type"] = "continuous"
master_config["addr"] = "0.0.0.0"
master_config["port"] = 5555
master_config["mailbox_buffer_size"] = 1 * 1024**3
master_config["min_num_minions"] = 1
master_config["max_num_minions"] = 2
master_config["minion_job_timeout"] = 3600.0
master_config["minion_params"] = {}
master_config["nn_format"] = "tflite"
master_config["nn_size"] = 64 * 1024**1
master_config["output_dir"] = "~/lexci_results"
master_config["b_save_training_data"] = False
master_config["b_save_sample_batches"] = False
master_config["validation_interval"] = 10
master_config["checkpoint_file"] = ""
master_config["model_h5_folder"] = ""
# If the documentation string isn't empty, the universal Master will create a
# text file called 'Documentation.txt' in the training's log directory and write
# the content of the string into said file.
master_config["doc"] = ""
# =============================================================================#


# PPO configuration dictionary
from ray.rllib.agents.ppo import DEFAULT_CONFIG

ppo_config = copy.deepcopy(DEFAULT_CONFIG)
# =========================== MAKE ADJUSTMENTS HERE ===========================#
ppo_config["model"]["fcnet_hiddens"] = [16, 16, 16]
ppo_config["model"]["fcnet_activation"] = "tanh"
ppo_config["train_batch_size"] = 10000  # = Number of experiences per cycle
ppo_config["sgd_minibatch_size"] = 64
ppo_config["num_sgd_iter"] = 8
ppo_config["gamma"] = 0.999
ppo_config["clip_param"] = 0.3
ppo_config["vf_clip_param"] = 1e6
ppo_config["lr"] = 1e-5
ppo_config["lr_schedule"] = [[0, 0.0025], [1e9, 1e-5]]
# =============================================================================#


if __name__ == "__main__":
    ConfigCreator(master_config, ppo_config, "ppo").run()
