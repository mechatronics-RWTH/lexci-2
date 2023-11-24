"""Functions for working with offline data.

File:   lexci2/utils/offline_data.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2023-02-02


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

from ray.rllib.policy.sample_batch import SampleBatch
import ray.rllib.offline.json_reader
import ray.rllib.offline.json_writer

import csv
import json
import re
import os
import logging
import numpy as np


logger = logging.getLogger(__name__)


def import_lexci_episode_csv(
    file_name: str, agent_id: str, b_import_aux_data=False
) -> Episode:
    """Read a LExCI training or validation run CSV and create an Episode object
    based on it.

    Arguments:
        - file_name: str
              Name of the CSV file to import.
        - agent_id: str
              Agent ID to attribute to the imported data.
        - b_import_aux_data: bool (Default: False)
              Whether to read auxiliary data as well.

    Returns:
        - _: Episode
              An episode containing the data that was imported from the CSV.
    """

    episode = Episode(agent_id)

    with open(file_name, "r") as f:
        reader = csv.DictReader(f, delimiter=";")

        # Sort the column names
        obs_cols = []
        action_cols = []
        new_obs_cols = []
        reward_col = []
        done_col = []
        t_col = []
        aux_data_cols = []
        for col in reader.fieldnames:
            if re.match(r"^obs\[\d+\]$", col) is not None:
                obs_cols.append(col)
            elif re.match(r"^action\[\d+\]$", col) is not None:
                action_cols.append(col)
            elif re.match(r"^new_obs\[\d+\]$", col) is not None:
                new_obs_cols.append(col)
            elif col == "reward":
                reward_col.append(col)
            elif col == "done":
                done_col.append(col)
            elif col == "t":
                t_col.append(col)
            elif b_import_aux_data:
                aux_data_cols.append(col)

        for row in reader:
            obs = np.array(
                [float(row[col]) for col in obs_cols], dtype=np.float32
            )
            action = np.array(
                [float(row[col]) for col in action_cols], dtype=np.float32
            )
            new_obs = np.array(
                [float(row[col]) for col in new_obs_cols], dtype=np.float32
            )
            reward = float(row[reward_col[0]])
            done = row[done_col[0]] == "True"
            t = float(row[t_col[0]]) if len(t_col) > 0 else None
            aux_data = {col: row[col] for col in aux_data_cols}

            experience = Experience(
                obs, action, new_obs, reward, done, t, aux_data
            )
            episode.append_experience(experience)

    return episode


def import_lexci_episode_csv_folder(
    folder_name: str, agent_id: str, b_import_aux_data: bool
) -> list[Episode]:
    """Import offline data by reading all training and validation run CSVs in a
    certain folder.

    This function doesn't search the specified folder recursively. All files
    that are to be imported must be on the top level. Also, they must abide by
    LExCI's naming convention for training/validation runs.

    Arguments:
        - folder_name: str
              Name of the folder to import from.
        - agent_id: str
              Agent ID to attribute to the imported data.
        - b_import_aux_data: bool (Default: False)
              Whether to read auxiliary data as well.

    Returns:
        - _: list[Episode]
              List of imported episodes.
    """

    episodes = []

    # Get a list of all CSV files that can be imported
    file_names_all = os.listdir(folder_name)
    file_names = []
    for e in file_names_all:
        m = re.match(r"^Cycle_\d+_(TrainingData|ValidationData)_\d+\.csv$", e)
        if m is not None:
            file_names.append(e)

    # Import the files
    for i, e in enumerate(file_names):
        csv_name = os.path.join(folder_name, e)
        episode = import_lexci_episode_csv(
            csv_name, agent_id, b_import_aux_data
        )
        logger.info(f"Imported {i + 1}/{len(file_names)} CSVs.")
        episodes.append(episode)

    return episodes


def import_sample_batch_json(file_name: str) -> SampleBatch:
    """Import a `SampleBatch` from a JSON file.

    Arguments:
        - file_name: str
              Name of the JSON file to import.

    Returns:
        - _: SampleBatch
              Imported `SampleBatch` object.
    """

    with open(file_name, "r") as f:
        d = json.load(f)
    return ray.rllib.offline.json_reader.from_json_data(d, None)


def export_sample_batch_json(sample_batch: SampleBatch, file_name: str) -> None:
    """Export a `SampleBatch` object as a JSON file.

    Arguments:
        - sample_batch: SampleBatch
              The `SampleBatch` object to export.
        - file_name: str
              Name of the JSON file to create.
    """

    json_string = ray.rllib.offline.json_writer._to_json(sample_batch, [])
    with open(file_name, "w") as f:
        f.write(json_string)


def import_sample_batch_json_folder(folder_name: str) -> list[SampleBatch]:
    """Import offline data by reading all training and validation run sample
    batch JSONs in a certain folder.

    This function doesn't search the specified folder recursively. All files
    that are to be imported must be on the top level. Also, they must abide by
    LExCI's naming convention for training/validation sample batch JSONs.

    Arguments:
        - folder_name: str
              Name of the folder to import from.

    Returns:
        - _: list[SampleBatch]
              List containing the imported `SampleBatch` objects.
    """

    episode_sample_batches = []

    # Get a list of all JSON files that can be imported
    file_names_all = os.listdir(folder_name)
    file_names = []
    for e in file_names_all:
        m = re.match(r"^Cycle_\d+_(Training|Validation)_SampleBatch\.json$", e)
        if m is not None:
            file_names.append(e)

    # Import the files
    for i, e in enumerate(file_names):
        json_name = os.path.join(folder_name, e)
        sample_batch = import_sample_batch_json(json_name)
        logger.info(f"Imported {i + 1}/{len(file_names)} sample batch JSONs.")
        episode_sample_batches.append(sample_batch)

    return episode_sample_batches
