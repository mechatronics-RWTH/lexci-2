"""A class for interacting with dSPACE ControlDesk for the purpose of generating
and collecting RL-data.

File:   minion/controllers/controldesk_controller.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2022-08-11


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


from lexci2.data_containers import Experience, Episode

from win32com.client import Dispatch

import os
import time
import logging
import numpy as np
from typing import Any, Optional


logger = logging.getLogger(__name__)


class ControlDeskController:
    """Controller for ControlDesk, an ECU experiment and instrumentation
    software by dSPACE.
    """

    def __init__(self) -> None:
        """Initialize the controller."""

        self._app = None

    def start_controldesk(
        self, project_file: str, experiment_name: str
    ) -> None:
        """Start ControlDesk and open a project and experiment and connect to
        the platform.

        If the application is already open, it is closed first and then
        re-started.

        Arguments:
            - project_file: str
                  The project file to open.
            - experiment_name: str
                  Name of the experiment.
        """

        if self._app is not None:
            self.stop_controldesk()

        self._app = Dispatch("ControlDeskNG.Application")
        self._app.OpenExperiment(project_file, experiment_name)
        self._app.ActiveExperiment.Platforms[0].Connect()
        self._app.CalibrationManagement.StartOnlineCalibration()

    def stop_controldesk(self) -> None:
        """Close ControlDesk and disconnect from the platform."""

        if self._app is not None:
            self._app.CalibrationManagement.StopOnlineCalibration()
            self._app.ActiveExperiment.Platforms[0].Disconnect()
            self._app.Quit()
            self._app = None
            time.sleep(10)

    def kill_controldesk(self) -> None:
        """Kill ControlDesk."""

        try:
            os.system("taskkill /f /im ControlDesk.exe")
            self._app = None
        except:
            pass

    def read_var(
        self, var_name: str, platform_idx: Optional[int] = None
    ) -> list[Any]:
        """Read a variable inside the loaded model.

        Arguments:
            - var_name: str
                  Name of the variable, including the full path.
            - platform_idx: Optional[int] (Default: None)
                  The platform index (i.e. starting at 0) of the variable's
                  model.

        Returns:
            - _: list[Any]
                  The value of the variable.
        """

        if platform_idx is None:
            plat = self._app.ActiveExperiment.Platforms[0]
        else:
            plat = self._app.ActiveExperiment.Platforms[0].Platforms[
                platform_idx
            ]

        v = plat.ActiveVariableDescription.Variables.Item(
            var_name
        ).ValueConverted
        return v if isinstance(v, list) else [v]

    def write_var(
        self, var_name, value: list[Any], platform_idx: Optional[int] = None
    ) -> None:
        """Write the value of a variable inside the loaded model.

        Arguments:
            - var_name: str
                  Name of the variable, including the full path.
            - value: list[Any]
                  Value to write.
            - platform_idx: Optional[int] (Default: None)
                  The platform index (i.e. starting at 0) of the variable's
                  model.
        """

        if platform_idx is None:
            plat = self._app.ActiveExperiment.Platforms[0]
        else:
            plat = self._app.ActiveExperiment.Platforms[0].Platforms[
                platform_idx
            ]

        var = plat.ActiveVariableDescription.Variables.Item(var_name)
        var.ValueConverted = value

    def start_triggered_recording(
        self, output_file: str, recorder_idx: Optional[int] = None
    ) -> None:
        """Start a triggered recording inside ControlDesk.

        Arguments:
            - output_file: str
                  Name of the output file.
            - recorder_idx: Optional[int] (Default: None)
                  Index of the recorder.
        """

        if recorder_idx is None:
            rec = self._app.MeasurementDataManagement.MainRecorder
        else:
            rec = self._app.MeasurementDataManagement.Recorders.Item(
                recorder_idx
            )

        rec.AutomaticNamingEnabled = False
        rec.ExportFullPath = output_file
        rec.Start(True, True)

    def stop_triggered_recording(
        self, recorder_idx: Optional[int] = None
    ) -> None:
        """Stop a triggered recording inside ControlDesk.

        Arguments:
            - recorder_idx: Optional[int] (Default: None)
                  Index of the recorder.
        """

        if recorder_idx is None:
            rec = self._app.MeasurementDataManagement.MainRecorder
        else:
            rec = self._app.MeasurementDataManagement.Recorders.Item(
                recorder_idx
            )

        rec.Stop()

    @staticmethod
    def _get_csv_sections(output_file: str) -> list[str]:
        """Generator method that yields the sections of a ControlDesk recorder
        CSV. In said CSVs, sections are delimited by empty lines.

        Arguments:
            - output_file: str
                  Name of the output CSV file.

        Yields:
            - _: list[str]
                  The lines of a section within the CSV file.
        """

        section = []
        with open(output_file, "r") as f:
            while True:
                l = f.readline()

                if l == "\n":
                    # Found a section delimiter
                    yield section
                    section.clear()
                elif l == "":
                    # Reached the end of the file
                    yield section
                    break
                else:
                    if l[-1] == "\n":
                        # Remove the trailing newline character
                        section.append(l[:-1])
                    else:
                        section.append(l)

    @staticmethod
    def _is_rl_block_section(section: list[str]) -> bool:
        """Determine whether a section of a ControlDesk recorder CSV contains
        the output of a Reinforcement Learning Block.

        Arguments:
            - section: list[str]
                  Lines of a section.

        Returns:
            - _: bool
                  `True` if the section contains data from an RL Block, else
                  `False`.
        """

        for l in section:
            if "RL_Agent" in l:
                return True

        return False

    @staticmethod
    def _extract_episode_from_section(section: list[str]) -> Episode:
        """Given the CSV section of an RL Block, extract LExCI experiences and
        put them into an episode.

        Arguments:
            - section: list[str]
                  Lines of a section.

        Returns:
            - _: Episode
                  A LExCI episode.
        """

        # Paths inside the Simulink model to the measured variables
        var_paths = []
        # Names of the measured variables
        var_names = []

        # Column indices of the observation variables
        obs_idxs = []
        # Column indices of the action variables
        action_idxs = []
        # Column indices of the new observation variables
        new_obs_idxs = []
        # Column index of the reward variable
        reward_idx = 0
        # Column index of the "done" flag variable
        done_idx = 0
        # Column indices of auxiliary data variables
        aux_data_idxs = {}
        # Column index of the execution trigger variable
        exec_trigger_idx = 0

        # Get the indices of the data that shall be read
        for l in section:
            if "trace_names" in l:
                var_names = l.split(",")
            elif "path" in l:
                var_paths = l.split(",")

                # Get the column indices of mandatory quantities
                obs_idxs = [
                    i
                    for i, e in enumerate(var_paths)
                    if "norm_observation_out" in e
                    and "new_norm_observation_out" not in e
                ]
                action_idxs = [
                    i for i, e in enumerate(var_paths) if "norm_action_out" in e
                ]
                new_obs_idxs = [
                    i
                    for i, e in enumerate(var_paths)
                    if "new_norm_observation_out" in e
                ]
                reward_idx = [
                    i for i, e in enumerate(var_paths) if "reward_out" in e
                ]
                reward_idx = reward_idx[0]  # There's only one reward
                done_idx = [
                    i
                    for i, e in enumerate(var_paths)
                    if "b_episode_finished_out" in e
                ]
                done_idx = done_idx[0]  # There's only one "done" flag
                exec_trigger_idx = [
                    i
                    for i, e in enumerate(var_paths)
                    if "execution_trigger_out" in e
                ]
                exec_trigger_idx = exec_trigger_idx[
                    0
                ]  # There's only one trigger

                # Get the column indices of auxiliary quantities
                for i in range(len(var_paths)):
                    if i not in [
                        *obs_idxs,
                        *action_idxs,
                        *new_obs_idxs,
                        reward_idx,
                        done_idx,
                        exec_trigger_idx,
                    ]:
                        aux_data_name = var_paths[i] + "/" + var_names[i]
                        aux_data_idxs[aux_data_name] = i
                break
        else:
            logger.warn(
                "Couldn't find the variable names or paths inside section."
            )
            return None

        # Find the line number where the actual data starts
        data_start_idx = [
            i for i, e in enumerate(section) if "trace_values" in e
        ]
        data_start_idx = data_start_idx[0]

        # Read the data
        episode = Episode("agent0")  # TODO: Set the proper ID!
        for l in section[data_start_idx:]:
            columns = l.split(",")

            # Read mandatory quantities
            obs = np.array(
                [float(columns[idx]) for idx in obs_idxs], dtype=np.float32
            )
            action = np.array(
                [float(columns[idx]) for idx in action_idxs], dtype=np.float32
            )
            new_obs = np.array(
                [float(columns[idx]) for idx in new_obs_idxs], dtype=np.float32
            )
            reward = float(columns[reward_idx])
            done = bool(int(columns[done_idx]))
            exec_trigger = bool(int(columns[exec_trigger_idx]))
            t = float(columns[1])

            if not exec_trigger:
                # The RL block wasn't actually executed so there is no new data
                continue

            # Read auxiliary quantities
            aux_data = {}
            for aux_name in aux_data_idxs:
                try:
                    aux_data[aux_name] = [
                        float(columns[aux_data_idxs[aux_name]])
                    ]
                except:
                    pass

            experience = Experience(
                obs, action, new_obs, reward, done, t, aux_data
            )
            episode.append_experience(experience)

        return episode

    def is_csv_ready(self, output_file: str, test_interval: float = 5) -> bool:
        """Check whether an output CSV file exists and is not being written to
        by another process.

        To test the latter, its size is determined twice using some waiting
        interval: If the resulting values are equal, it is assumed that the
        output file is not being written to. This step is necessary because
        ControlDesk generates rather large CSVs that take some time to be
        written to the disk.

        Arguments:
            - output_file: str
                  Name of the output CSV file.
            - test_interval: float (Default: 5, Unit: s)
                  Time between getting the file size.

        Returns:
            - _: bool
                  `True` if the CSV file is ready, else `False`.
        """

        try:
            with open(output_file, "r") as f:
                # Move the handle to the end of the file and store its position
                # as that value is equal to the file size in bytes
                f.seek(0, 2)
                size_1 = f.tell()

            time.sleep(test_interval)

            with open(output_file, "r") as f:
                # Move the handle to the end of the file and store its position
                # as that value is equal to the file size in bytes
                f.seek(0, 2)
                size_2 = f.tell()

            if size_1 == size_2:
                return True
            else:
                return False
        except:
            return False

    def extract_csv_data(
        self, output_file: str, delete_csv: bool = False
    ) -> list[Episode]:
        """Extract the RL Block's experiences from a CSV generated by a
        ControlDesk recorder.

        Arguments:
            - output_file: str
                  Name of the output CSV file.
            - delete_csv: bool (Default: False)
                  Whether to delete the CSV file once it has been processed.

        Returns:
            - _: list[Episode]
                  List of LExCI episodes. There's one episode for every RL Block
                  in the model.
        """

        # Extract episodes
        episodes = []
        for section in ControlDeskController._get_csv_sections(output_file):
            if ControlDeskController._is_rl_block_section(section):
                episode = ControlDeskController._extract_episode_from_section(
                    section
                )
                episodes.append(episode)

        # Delete the output file if the user so wishes
        if delete_csv and self.is_csv_ready(output_file):
            # The CSV file is only deleted if it still exists and isn't being
            # written to by another process
            os.remove(output_file)

        return episodes
