"""A class for interacting with TraceTronic's ECU-TEST for the purpose of
generating and collecting RL data.

File:   lexci2/minion/controllers/ecu_test_controller.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2022-09-07


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
from ApiClient import ApiClient

import os
import time
import datetime
import json
import logging
from typing import Any, Optional


logger = logging.getLogger(__name__)


class EcuTestController:
    """Controller for ECU-TEST, a test automation software by TraceTronic."""

    def __init__(self, ecu_test_exe_path: str) -> None:
        """Initialize the controller.

        Arguments:
            - ecu_test_exe_path: str
                  Absolute path to ECU-TEST's executable.
        """

        self._ecu_test_exe_path = ecu_test_exe_path

        self._api = ApiClient()

        self._app = None
        self._test_env = None

    def start_ecu_test(self) -> None:
        """Start ECU-TEST.

        ECU-TEST must be configured such that it doesn't ask for the workspace
        upon startup. Instead, it must remember it. You can achieve that by
        opening the program manually, navigating to the workspace directory, and
        then closing ECU-TEST.
        """

        os.system(f'"{self._ecu_test_exe_path}"')
        self._app = Dispatch("ECU-TEST.Application")
        self._test_env = self._app.Start()

    def close_ecu_test(self) -> None:
        """Close ECU-TEST."""

        self._app.Quit()
        self._app = None
        self._test_env = None

    def kill_ecu_test(self) -> None:
        """Kill ECU-TEST the programs it controls."""

        try:
            # ECU-TEST
            os.system("taskkill /f /im ECU-TEST.exe")
            self._app = None
            self._test_env = None

            # Controlled programs
            os.system("taskkill /f /im ControlDesk.exe")
            os.system("taskkill /f /im Inca.exe")
        except:
            pass

    def kill_controldesk(self) -> None:
        """Kill ControlDesk."""

        try:
            os.system("taskkill /f /im ControlDesk.exe")
        except:
            pass

    def kill_inca(self) -> None:
        """Kill INCA."""

        try:
            os.system("taskkill /f /im Inca.exe")
        except:
            pass

    def execute_package(
        self, pckg_file: str, timeout: Optional[float] = None
    ) -> None:
        """Execute an ECU-TEST package.

        Arguments:
            - pckg_file: str
                  ECU-TEST package to execute.
            - timeout: Optional[float] (Default: None, Unit: s)
                  Time in seconds after which the execution shall time out.

        Raises:
            - An `EcuTestControllerExecutionTimeoutError` is thrown if `timeout`
              has been set and execution is taking longer than that.
        """

        t_start = datetime.datetime.now()
        self._app.OpenPackage(pckg_file)
        exec_info = self._test_env.ExecutePackage(pckg_file)
        while exec_info.GetState() == "RUNNING":
            # Check whether the execution is taking too long
            if timeout is not None:
                if (
                    datetime.datetime.now() - t_start
                ).total_seconds() >= timeout:
                    raise EcuTestControllerExecutionTimeoutError(
                        "Execution of package" f" '{pckg_file}' timed out."
                    )
            time.sleep(0.5)


class EcuTestControllerExecutionTimeoutError(RuntimeError):
    """Exception which indicates that the execution of an ECU-TEST package has
    timed out.
    """

    pass
