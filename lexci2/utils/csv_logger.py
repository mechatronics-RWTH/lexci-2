"""Class for logging LExCI's training metrics in a CSV file.

File:   utils/csv_logger.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2022-11-02


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


from lexci2.data_containers import Experience, Episode, Cycle  # TODO

import csv
import datetime


class CsvLogger:
    """Log LExCI's training metrics in a CSV file."""

    def __init__(self, file_name: str) -> None:
        """Initialize the logger.

        Arguments:
            - file_name: str
                  Name of the CSV log file.
        """

        col_names = [
            "timestamp",
            "cycle_no",
            "episode_reward_max",
            "episode_reward_min",
            "episode_reward_mean",
            "num_episodes",
        ]
        self._file = open(file_name, "w")
        self._dict_writer = csv.DictWriter(
            self._file, fieldnames=col_names, delimiter=";"
        )
        self._dict_writer.writeheader()

    def close(self) -> None:
        """Close the logger and its file."""

        if self._file is not None:
            self._file.close()
            self._file = None

    def write(self, cycle: Cycle, cycle_no: int) -> None:
        """Analyze a LExCI cycle and write its metrics to the CSV log file.

        Arguments:
            - cycle: Cycle
                  LExCI cycle data to log.
            - cycle_no: int
                  Current LExCI cycle number.
        """

        # Collect data
        episode_rewards = []
        for episode in cycle.eps:
            cum_reward = 0
            for experience in episode.exps:
                cum_reward += experience.reward
            episode_rewards.append(cum_reward)

        # Write the metrics
        d = {
            "timestamp": str(datetime.datetime.now()),
            "cycle_no": cycle_no,
            "episode_reward_max": max(episode_rewards),
            "episode_reward_min": min(episode_rewards),
            "episode_reward_mean": sum(episode_rewards) / len(episode_rewards),
            "num_episodes": cycle.get_num_episodes(),
        }
        self._dict_writer.writerow(d)
        self._file.flush()
