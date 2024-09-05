"""Helper class for reading LExCI's CSV log files while they're being written
to.

File:   lexci2/utils/csv_log_reader.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2024-08-28


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

import csv
from typing import Any


class CsvLogReader:
    """Helper class for reading data from LExCI's CSV log files while they are
    being written to."""

    def __init__(self, csv_file_name: str) -> None:
        """Initialize the reader.

        Arguments:
            - csv_file_name: str
                  Name of the CSV log file to read.
        """

        self._csv_file_name = csv_file_name
        self._csv_rows = None

        self._update()

    def _update(self) -> None:
        """Synchronize the data stored in the object with the current state of
        the CSV file.
        """

        with open(self._csv_file_name, "r") as f:
            self._csv_rows = list(csv.DictReader(f, delimiter=";"))

    def _get_row(self, cycle_no: int) -> dict[str, Any]:
        """Synchronize the object with the CSV file on disk and get the data row
        for a given LExCI cycle.

        Arguments:
            - cycle_no: int
                  The cycle number for which to get the value.

        Returns:
            - _: dict[str, Any]
                  The data row of the cycle number.

        Raises:
            - ValueError:
                  - If the cycle number doesn't exist.
        """

        self._update()

        # Check the cycle number
        if not (0 <= cycle_no < len(self._csv_rows)):
            raise ValueError(
                f"The cycle number {cycle_no} doesn't (currently) exist."
            )

        return self._csv_rows[cycle_no]

    def get_num_cycles(self) -> int:
        """Get the number of cycles in the log file.

        Returns:
            - _: int
                  The number of cycles recorded in the CSV log file.
        """

        self._update()
        return len(self._csv_rows)

    def get_timestamp(self, cycle_no: int) -> str:
        """Get the timestamp of a cycle number.

        Arguments:
            - cycle_no: int
                  The cycle number for which to get the value.

        Returns:
            - _: str
                  The timestamp of the cycle.
        """

        return str(self._get_row(cycle_no)["timestamp"])

    def get_episode_reward_max(self, cycle_no: int) -> float:
        """Get the maximum episode reward in a given LExCI cycle.

        Arguments:
            - cycle_no: int
                  The cycle number for which to get the value.

        Returns:
            - _: float
                  The maximum episode reward of the cycle.
        """

        return float(self._get_row(cycle_no)["episode_reward_max"])

    def get_episode_reward_min(self, cycle_no: int) -> float:
        """Get the minimum episode reward in a given LExCI cycle.

        Arguments:
            - cycle_no: int
                  The cycle number for which to get the value.

        Returns:
            - _: float
                  The minimum episode reward of the cycle.
        """

        return float(self._get_row(cycle_no)["episode_reward_min"])

    def get_episode_reward_mean(self, cycle_no: int) -> float:
        """Get the mean episode reward in a given LExCI cycle.

        Arguments:
            - cycle_no: int
                  The cycle number for which to get the value.

        Returns:
            - _: float
                  The mean episode reward of the cycle.
        """

        return float(self._get_row(cycle_no)["episode_reward_mean"])

    def get_num_episodes(self, cycle_no: int) -> int:
        """Get the number of episodes in a given LExCI cycle.

        Arguments:
            - cycle_no: int
                  The cycle number for which to get the value.

        Returns:
            - _: int
                  The number of episodes in the cycle.
        """

        return int(self._get_row(cycle_no)["num_episodes"])

    def get_timestamp_list(self) -> list[str]:
        """Get the list of all timestamps in chronological order.

        Arguments:
            - cycle_no: int
                  The cycle number for which to get the value.

        Returns:
            - _: list[str]
                  The list of timestamps.
        """

        self._update()
        return [str(e["timestamp"]) for e in self._csv_rows]

    def get_episode_reward_max_list(self) -> list[float]:
        """Get the list of all maximum episode rewards in chronological order.

        Returns:
            - _: list[float]
                  The list of maximum episode rewards.
        """

        self._update()
        return [float(e["episode_reward_max"]) for e in self._csv_rows]

    def get_episode_reward_min_list(self) -> list[float]:
        """Get the list of all minimum episode rewards in chronological order.

        Returns:
            - _: list[float]
                  The list of minimum episode rewards.
        """

        self._update()
        return [float(e["episode_reward_min"]) for e in self._csv_rows]

    def get_episode_reward_mean_list(self) -> list[float]:
        """Get the list of all mean episode rewards in chronological order.

        Returns:
            - _: list[float]
                  The list of mean episode rewards.
        """

        self._update()
        return [float(e["episode_reward_mean"]) for e in self._csv_rows]

    def get_num_episodes_list(self) -> list[int]:
        """Get the list of all numbers of episodes in chronological order.

        Returns:
            - _: list[int]
                  The list of episode numbers.
        """

        self._update()
        return [int(e["num_episodes"]) for e in self._csv_rows]
