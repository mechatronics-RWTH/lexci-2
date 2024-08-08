"""Test LExCI's setup procedure. It's assumed that all required build tools
(e.g. `gcc`, `gpp`, `g++`, etc.) are already installed. Please consult the
documentation for more details.

File:   tests/system/test_installation.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2024-08-07


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

import unittest
import subprocess
import tempfile
import os


# The top-level directory of the repository
TOP_LEVEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)


class TestInstallation(unittest.TestCase):
    """Test LExCI's setup procedure."""

    def test_installation(self) -> None:
        """Install LExCI and check whether there are any failures."""

        # Check whether Python 3.9.15 is installed
        with subprocess.Popen(
            "/bin/bash",
            shell=True,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ) as proc:
            outputs, _ = proc.communicate("python3.9 --version")
            self.assertEqual(
                outputs,
                "Python 3.9.15\n",
                "Python 3.9.15 is not installed on the system.",
            )

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            # Create a temporary virtual environment
            with subprocess.Popen(
                "/bin/bash",
                shell=True,
                text=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ) as proc:
                venv_dir_name = os.path.join(tmp_dir_name, ".venv/lexci2")
                cmd = f"python3.9 -m venv {venv_dir_name}"
                proc.communicate(cmd)
                self.assertEqual(
                    proc.returncode,
                    0,
                    "Failed to create the virtual environment.",
                )
                venv_activation_script = os.path.abspath(
                    os.path.join(venv_dir_name, "bin/activate")
                )

            # Install pip
            with subprocess.Popen(
                "/bin/bash",
                shell=True,
                text=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ) as proc:
                cmd = f"source {venv_activation_script}"
                cmd += " && python3.9 -m pip install pip==22.0.4"
                proc.communicate(cmd)
                self.assertEqual(
                    proc.returncode,
                    0,
                    "Couldn't install the required version of pip.",
                )

            # Install/downgrade certain software packages
            with subprocess.Popen(
                "/bin/bash",
                shell=True,
                text=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ) as proc:
                cmd = f"source {venv_activation_script}"
                cmd += (
                    " && pip install setuptools==58.1.0 wheel==0.38.4"
                    + " numpy==1.26.4"
                )
                proc.communicate(cmd)
                self.assertEqual(
                    proc.returncode,
                    0,
                    (
                        "Couldn't install/downgrade to the required versions of"
                        + " setuptools, wheel, and numpy."
                    ),
                )

            # Install LExCI
            with subprocess.Popen(
                "/bin/bash",
                shell=True,
                text=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ) as proc:
                cmd = f"source {venv_activation_script}"
                cmd += f" && cd {TOP_LEVEL_DIR}"
                cmd += " && pip install ."
                proc.communicate(cmd)
                self.assertEqual(proc.returncode, 0, "Failed to install LExCI.")


if __name__ == "__main__":
    unittest.main()
