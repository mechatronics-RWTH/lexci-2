"""A class for interacting with dSPACE VEOS Player.

File:   minion/controllers/veos_player_controller.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2022-08-30


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


from win32com.client import Dispatch

import os
import logging
import time


logger = logging.getLogger(__name__)


class VeosPlayerController:
    """Controller for dSPACE VEOS Player."""

    def __init__(self) -> None:
        """Initialize the controller."""

        self._app = None
        self._proj = None

    def start_veos_player(self, project_file: str) -> None:
        """Start VEOS Player, open a project, and download it."""

        # Start VEOS Player
        self._app = Dispatch("VeosPlayer.Application")
        self._proj = self._app.Projects.Open(project_file)
        self._app.Simulator.Download()

    def close_veos_player(self) -> None:
        """Quit VEOS Player."""

        if self._proj is not None:
            self._proj.Close(saveChanges=False)
            self._proj = None
            time.sleep(5)
        if self._app is not None:
            self._app.Quit()
            self._app = None
            time.sleep(10)

    def kill_veos_player(self) -> None:
        """Kill VEOS Player and the VEOS kernel."""

        try:
            os.system("taskkill /f /im VeosPlayer.exe")
            os.system("taskkill /f /im VeosKernel.exe")
            self._app = None
            self._proj = None
        except:
            pass

    def start_simulation(self) -> None:
        """Start the simulation."""

        self._app.Simulator.Start()

    def stop_simulation(self) -> None:
        """Stop the simulation."""

        if self._app is not None and self._app.Simulator is not None:
            self._app.Simulator.Stop()

    def is_kernel_running(self) -> bool:
        """Check whether the VEOS kernel is running.

        Returns:
            - _: bool
                  `True` if the kernel is running, otherwise `False`.
        """

        try:
            b_status = self._app.Simulator.SimulatorState == "Running"
            return b_status
        except:
            return False
