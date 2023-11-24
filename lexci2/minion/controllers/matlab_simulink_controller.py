"""A class for interacting with MATLAB/Simulink for the purpose of generating
and collecting RL-data.

For installing the modules required by this file, please refer to:
https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

WARNING: MATLAB's `load_system()` method (and possibly some others as well) does
not work with certain Ray/RLlib imports (e.g.
`from ray.rllib.env.external_env import ExternalEnv`).

File:   minion/controllers/matlab_simulink_controller.py
Author: Tobias Brinkmann (brinkmann@mmp.rwth-aachen.de)
        Lucas Koch (koch_luc@mmp.rwth-aachen.de)
        Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2023-09-18


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

import matlab
import matlab.engine
import logging

from typing import Any, Optional


logger = logging.getLogger(__name__)


class MatlabSimulinkController:
    """A controller for MATLAB/Simulink."""

    def __init__(self) -> None:
        """Initialize the controller."""

        self._app = None
        self._model_folder = None
        self._model_name = None

    def start_matlab_simulink(
        self,
        model_folder: str,
        model_name: str,
        init_script_name: str,
        open_gui: bool = False,
    ) -> None:
        """Start MATLAB, run the initialization script, and open the Simulink
        model.

        Arguments:
            - model_folder: str
                  Directory where the model and all of its data files are
                  located.
            - model_name: str
                  Name of the Simulink model.
            - init_script_name: str
                  Name of the MATLAB script that shall be run right after MATLAB
                  started.
            - open_gui: bool (Default: False)
                  Whether to open MATLAB's GUI.
        """

        # Copy arguments
        self._model_folder = model_folder
        self._model_name = model_name

        # Open MATLAB
        matlab_args = "-desktop" if open_gui else ""
        self._app = matlab.engine.start_matlab(matlab_args)

        # Change the directory
        self.run_cmd(f"cd {self._model_folder}")
        # Recursively add the model directory to the search path
        self.run_cmd(f"addpath(genpath('{self._model_folder}'))")
        # Execute the initialization script
        self.run_cmd(init_script_name)
        # Open the Simulink model
        self.run_cmd(f"load_system('{self._model_name}')")

    def stop_matlab_simulink(self) -> None:
        """Close MATLAB/Simulink."""

        if self._app is not None:
            self._app.quit()
            self._app = None
            self._model_folder = None
            self._model_name = None

    def read_workspace_var(self, var_name: str) -> Any:
        """Read a variable from the MATLAB workspace.

        Arguments:
            - var_name: str
                  Name of the MATLAB workspace variable.

        Returns:
            - _: Any
                  The value of the MATLAB workspace variable.
        """

        return self._app.workspace[var_name]

    def write_workspace_var(
        self, var_name: str, value: Any, dtype: Optional[type] = None
    ) -> None:
        """Write a variable in the MATLAB workspace.

        Arguments:
            - var_name: str
                  Name of the MATLAB workspace variable.
            - value: Any
                  The value to write.
            - dtype: Optional[type] (Default: None)
                  If this is not `None`, the value is first cast to this data
                  type before writing it.
        """

        # Convert the data type if explicitly specified
        val = value
        if dtype is not None:
            val = dtype(val)

        self._app.workspace[var_name] = val

    def read_simulink_var(self, var_name: str) -> Any:
        """Read a variable from the Simulink model.

        Arguments:
            - var_name: str
                  Name of the Simulink variable, i.e. something like
                  'ModelName/Path/to/System/VariableName'. You can copy this
                  from Simulink's address bar and just add a slash and the
                  actual name of the variable.

        Returns:
            - _: Any
                  The value of the Simulink variable.
        """

        return self.read_simulink_block_param(var_name, "Value")

    def write_simulink_var(self, var_name: str, value: Any) -> None:
        """Write a variable in the Simulink model.

        Arguments:
            - var_name: str
                  Name of the Simulink variable, i.e. something like
                  'ModelName/Path/to/System/VariableName'. You can copy this
                  from Simulink's address bar and just add a slash and the
                  actual name of the variable.
            - value: Any
                  The value to write.
        """

        self.write_simulink_block_param(var_name, "Value", value)

    def read_simulink_block_param(self, block_name: str, param_name) -> Any:
        """Read a parameter of a Simulink block.

        Arguments:
            - block_name: str
                  Name of the Simulink block, i.e. something like
                  'ModelName/Path/to/System/BlockName'. You can copy this from
                  Simulink's address bar and just add a slash and the actual
                  name of the block.
            - param_name: str
                  Name of the parameter to read (e.g. 'Value' or 'Seed').

        Returns:
            - _: Any
                  The value of the block's parameter.
        """

        return self._app.get_param(block_name, param_name)

    def write_simulink_block_param(
        self, block_name: str, param_name: str, value: Any
    ) -> None:
        """Write a parameter of a Simulink block.

        Arguments:
            - block_name: str
                  Name of the Simulink block, i.e. something like
                  'ModelName/Path/to/System/BlockName'. You can copy this from
                  Simulink's address bar and just add a slash and the actual
                  name of the block.
            - param_name: str
                  Name of the parameter to read (e.g. 'Value' or 'Seed').
            - value: Any
                  The value to write.
        """

        self._app.set_param(block_name, param_name, str(value), nargout=0)

    def run_cmd(self, cmd: str, nargout: int = 0) -> Any:
        """Run a command in MATLAB's terminal.

        Arguments:
            - cmd: str
                  The command to execute.
            - nargout: int (Default: 0)
                  The number of return values of the command.
        """

        return self._app.eval(cmd, nargout=nargout)
