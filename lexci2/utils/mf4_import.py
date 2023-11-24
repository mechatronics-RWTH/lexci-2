"""Helper functions for reading MF4 files that contain LExCI experiences.

File:   lexci2/utils/mf4_import.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2022-10-07


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
from lexci2.data_containers import Experience, Episode

from asammdf import MDF
import numpy as np


def import_episode_mf4(
    file_name: str,
    agent_id: str,
    obs_sig_name: str,
    action_sig_name: str,
    new_obs_sig_name: str,
    reward_sig_name: str,
    done_sig_name: str,
) -> Episode:
    """Read an MF4 file which contains a LExCI episode and turn it into an
    `Episode` object.

    If you don't know the signal names, exporting the MF4 to a CSV file and
    looking in there might help.

    Arguments:
        - file_name: str
              Name of the MF4 file to import.
        - agent_id: str
              ID of the agent that was used to generate the data.
        - obs_sig_name: str
              Name of the signal that represents the agent's observation.
        - action_sig_name: str
              Name of the signal that represents the agent's action.
        - new_obs_sig_name: str
              Name of the signal that represents the agent's new observation.
        - reward_sig_name: str
              Name of the signal that represents the reward.
        - done_sig_name: str
              Name of the signal that represents the end-of-episode flag.

    Returns:
        - _: Episode
              An `Episode` object.
    """

    # Load the MF4 file
    mdf = MDF(file_name)

    # Select all required signals in the MF4
    obs_sig = mdf.select([obs_sig_name])[0]
    action_sig = mdf.select([action_sig_name])[0]
    new_obs_sig = mdf.select([new_obs_sig_name])[0]
    reward_sig = mdf.select([reward_sig_name])[0]
    done_sig = mdf.select([done_sig_name])[0]

    aux_data_sigs = {}
    for signal in mdf.iter_channels():
        if signal.name not in [
            obs_sig_name,
            action_sig_name,
            new_obs_sig_name,
            reward_sig_name,
            done_sig_name,
        ]:
            aux_data_sigs[signal.name] = mdf.select([signal.name])[0]

    # Read the experiences
    episode = Episode(agent_id)
    for i in range(len(obs_sig.samples)):
        # Mandatory information
        obs = np.array(obs_sig.samples[i][0], dtype=np.float32)
        action = np.array(
            [action_sig.samples[i]], dtype=np.float32
        )  # TODO: This may have to be adapted if the object actually is a list/array.
        new_obs = np.array(new_obs_sig.samples[i][0], dtype=np.float32)
        reward = float(reward_sig.samples[i])
        done = bool(done_sig.samples[i])

        # Time [s]
        t = float(obs_sig.timestamps[i])

        # Auxiliary data
        aux_data = {}
        for k, v in aux_data_sigs.items():
            try:
                aux_data[k] = v.samples[i]
            except:
                pass

        exp = Experience(obs, action, new_obs, reward, done, t, aux_data)
        episode.append_experience(exp)

    return episode


def convert_mf4_to_csv(mf4_file_name: str, csv_file_name: str) -> None:
    """Convert an MF4 file to one or more CSVs.

    Arguments:
        - mf4_file_name: str
              Name of the MF4 file to convert.
        - csv_file_name: str
              Name/prefix of the output file(s).
    """

    mdf = MDF(mf4_file_name)
    mdf.export("csv", csv_file_name)
