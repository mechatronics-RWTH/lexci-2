"""Train agents in the pendulum environment and check whether they converge.

File:   tests/system/test_training.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2024-08-26


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

from test_installation import install_lexci

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


class TestTraining(unittest.TestCase):
    """Train agents in the pendulum environment using various RL algorithms and
    check whether they converge to an optimum.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Create a temporary installation of the framework."""

        # The top-level directory of the repository
        top_level_dir_name = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        
        # Install LExCI in a temporary folder
        logger.info("Performing a temporary installation of LExCI...")
        cls._tmp_dir = tempfile.TemporaryDirectory()
        install_lexci(top_level_dir_name, cls._tmp_dir.name)
        logger.info("... done.")

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove the temporary installation of LExCI."""

        if cls._tmp_dir is not None:
            logger.info("Removing the temporary installation of LExCI...")
            cls._tmp_dir.cleanup()
            cls._tmp_dir = None
            logger.info("... done.")

    def test_ppo(self) -> None:
        """Train with a PPO agent."""

        # TODO
        pass

    def test_ddpg(self) -> None:
        """Train with a DDPG agent."""

        # TODO
        pass

    def test_td3(self) -> None:
        """Train with a TD3 agent."""

        # TODO
        pass


if __name__ == "__main__":
    unittest.main()
