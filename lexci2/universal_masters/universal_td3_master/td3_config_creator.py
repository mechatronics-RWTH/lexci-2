"""Script for creating the universal TD3 master's configuration file.

File:   lexci2/universal_masters/universal_td3_master/td3_config_creator.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2024-08-08


Copyright 2024 Teaching and Research Area Mechatronics in Mobile Propulsion,
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
master_config["addr"] = "0.0.0.0"
master_config["port"] = 5555
master_config["num_experiences_per_cycle"] = 1408  # = 128 * (10 + 1)
master_config["mailbox_buffer_size"] = 1 * 1024**3
master_config["min_num_minions"] = 1
master_config["max_num_minions"] = 2
master_config["minion_job_timeout"] = 3600.0
master_config["minion_params"] = {
    "INITIAL_STDEV": 0.5,
    "STDEV_DECAY_FACTOR": 0.925,
}
master_config["nn_format"] = "tflite"
master_config["nn_size"] = 64 * 1024**1
master_config["output_dir"] = "~/lexci_results"
master_config["b_save_training_data"] = False
master_config["b_save_sample_batches"] = False
master_config["validation_interval"] = 10
master_config["num_replay_trainings"] = 10
master_config["perc_replay_trainings"] = 0
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
# =============================================================================#


# TD3 configuration dictionary
from ray.rllib.agents.ddpg.td3 import TD3_DEFAULT_CONFIG

td3_config = copy.deepcopy(TD3_DEFAULT_CONFIG)
# =========================== MAKE ADJUSTMENTS HERE ===========================#
td3_config["actor_hiddens"] = [64, 64]
td3_config["actor_hidden_activation"] = "relu"
td3_config["critic_hiddens"] = [64, 64]
td3_config["critic_hidden_activation"] = "relu"
td3_config["replay_buffer_config"]["capacity"] = 10000
td3_config["store_buffer_in_checkpoints"] = True
td3_config["train_batch_size"] = 128
td3_config["gamma"] = 0.95
td3_config["actor_lr"] = 1e-3
td3_config["critic_lr"] = 1e-3
# Update target networks using `tau*policy + (1 - tau)*target_policy`
td3_config["tau"] = 0.01
td3_config["target_network_update_freq"] = 0
td3_config["l2_reg"] = 0.0
td3_config["policy_delay"] = 2
td3_config["target_noise"] = 0.2
td3_config["target_noise_clip"] = 0.5
# =============================================================================#


if __name__ == "__main__":
    ConfigCreator(master_config, td3_config, "td3").run()