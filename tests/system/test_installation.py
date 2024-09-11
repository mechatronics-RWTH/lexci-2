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
import logging


# Create the logger
logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
)
logger = logging.getLogger()


def install_lexci(top_level_dir_name: str, installation_dir_name: str) -> None:
    """Install LExCI in a virtual environment.

    This method can also be called by other test cases that require a full
    installation of the framework.

    Arguments:
        - top_level_dir_name: str
              Path to the top-level directory of the LExCI repository.
        - installation_dir_name: str
              Path to the folder where the virtual environment shall be created.
              Consequently, the folder will also contain the installation of
              LExCI.

    Raises:
        - RuntimeError:
              - If something goes wrong during the installation process.
    """

    # Check whether Python 3.9.15 is installed
    logger.info("Checking whether Python 3.9.15 is installed...")
    with subprocess.Popen(
        "/bin/bash",
        shell=True,
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ) as proc:
        outputs, _ = proc.communicate("python3.9 --version")
        if outputs != "Python 3.9.15\n":
            logger.info("... failed.")
            raise RuntimeError("Python 3.9.15 is not installed on the system.")
    logger.info("... done.")

    # Create the virtual environment
    logger.info("Creating LExCI's virtual environment...")
    with subprocess.Popen(
        "/bin/bash",
        shell=True,
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ) as proc:
        venv_dir_name = os.path.join(installation_dir_name, ".venv/lexci2")
        cmd = f"python3.9 -m venv {venv_dir_name}"
        proc.communicate(cmd)
        if proc.returncode != 0:
            logger.info("... failed.")
            raise RuntimeError("Failed to create the virtual environment.")
        venv_activation_script = os.path.abspath(
            os.path.join(venv_dir_name, "bin/activate")
        )
    logger.info("... done.")

    # Install pip
    logger.info("Installing the required version of `pip`...")
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
        if proc.returncode != 0:
            logger.info("... failed.")
            raise RuntimeError("Couldn't install the required version of pip.")
    logger.info("... done.")

    # Install/downgrade certain software packages
    logger.info(
        "Installing the required versions of `setuptools`, `wheel`, and `numpy`"
        + "..."
    )
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
            " && python3.9 -m pip install setuptools==58.1.0 wheel==0.38.4"
            + " numpy==1.26.4"
        )
        proc.communicate(cmd)
        if proc.returncode != 0:
            logger.info("... failed.")
            raise RuntimeError(
                "Couldn't install/downgrade to the required versions of"
                + " setuptools, wheel, and numpy."
            )
    logger.info("... done.")

    # Install LExCI
    logger.info("Installing LExCI...")
    with subprocess.Popen(
        "/bin/bash",
        shell=True,
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=top_level_dir_name,
    ) as proc:
        cmd = f"source {venv_activation_script}"
        cmd += " && python3.9 -m pip install ."
        proc.communicate(cmd)
        if proc.returncode != 0:
            logger.info("... failed.")
            raise RuntimeError("Failed to install LExCI.")
    logger.info("... done.")


class TestInstallation(unittest.TestCase):
    """Test LExCI's setup procedure."""

    def test_installation(self) -> None:
        """Install LExCI and check whether there are any failures."""

        # The top-level directory of the repository
        top_level_dir_name = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )

        # Install LExCI in a temporary folder
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            install_lexci(top_level_dir_name, tmp_dir_name)


if __name__ == "__main__":
    unittest.main()
