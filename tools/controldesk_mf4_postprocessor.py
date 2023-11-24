"""Postprocessing script for ControlDesk recorder MF4s.

File:   controldesk_mf4_postprocessor.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2023-03-10


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


from lexci2.utils.mf4_import import import_episode_mf4
from lexci2.utils.data_postprocessing import downsample_episode
from lexci2.utils.csv_export import export_episode_csv

import sys
import argparse


# Parse command line arguments
arg_parser = argparse.ArgumentParser(
    description=("Postprocessing script for" " ControlDesk recorder MF4s.")
)
arg_parser.add_argument(
    "controldesk_mf4", type=str, help=("ControlDesk" " recorder MF4 file.")
)
arg_parser.add_argument(
    "output_csv", type=str, help=("Postprocessed output CSV" " file.")
)
arg_parser.add_argument(
    "--target_step_length",
    type=float,
    help=(
        "Target step length to downsample to [s] (default: -1). A value of -1"
        + " means no sampling."
    ),
    default=-1.0,
)
cli_args = arg_parser.parse_args(sys.argv[1:])


episode = import_episode_mf4(
    file_name=cli_args.controldesk_mf4,
    agent_id="",
    obs_sig_name=r"Model Root/RiL_Pt/PPO_RL_Block/RL_Agent_Block/Experience Buffer/norm_observation_out/Out1",
    action_sig_name=r"Model Root/RiL_Pt/PPO_RL_Block/RL_Agent_Block/Experience Buffer/norm_action_out/Out1",
    new_obs_sig_name=r"Model Root/RiL_Pt/PPO_RL_Block/RL_Agent_Block/Experience Buffer/new_norm_observation_out/Out1",
    reward_sig_name=r"Model Root/RiL_Pt/PPO_RL_Block/RL_Agent_Block/Experience Buffer/reward_out/Out1",
    done_sig_name=r"Model Root/RiL_Pt/PPO_RL_Block/RL_Agent_Block/Experience Buffer/b_episode_finished_out/Out1",
)
if cli_args.target_step_length > 0:
    episode = downsample_episode(episode, cli_args.target_step_length)
export_episode_csv(episode, cli_args.output_csv)
