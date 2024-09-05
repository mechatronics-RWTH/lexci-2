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

from lexci2.utils.file_system import (
    find_newest_folder,
    find_newest_file,
    list_files,
)
from lexci2.utils.csv_log_reader import CsvLogReader
from lexci2.utils.math import apply_moving_average, calc_rmse

import os
import unittest
import subprocess
import tempfile
import time
import datetime
import csv
import shutil
import yaml
import numpy as np
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
        """Create a temporary folder for configuration files and results."""

        # The top-level directory of the repository
        cls._top_level_dir_name = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )

        # Create a temporary folder
        logger.info("Creating a temporary directory for the tests...")
        cls._tmp_dir = tempfile.TemporaryDirectory()
        logger.info("... done.")

        # Command for activating the virtual environment
        cls._venv_activation_cmd = os.environ.get(
            "LEXCI_VENV_ACTIVATION_CMD", "true"
        )

        # Constants
        cls._MOVING_AVERAGE_KERNEL_SIZE = 11
        cls._MAX_ALLOWED_RMSE = 300
        cls._MIN_SUCCESSFUL_VALIDATION_RETURN = -400

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove the temporary folder."""

        if cls._tmp_dir is not None:
            logger.info("Removing the temporary directory...")
            cls._tmp_dir.cleanup()
            cls._tmp_dir = None
            logger.info("... done.")

    def _run_training_test(
        self,
        algorithm: str,
        timeout: float,
        num_training_cycles: int,
    ) -> None:
        """Run a training and check whether it passes all required tests..

        Arguments:
            - algorithm: str
                  The RL algorithm to train with. This must be either `ppo`,
                  `ddpg`, or `td3`.
            - timeout: float (Unit: s)
                  Time after which the training times out.
            - num_training_cycles: int
                  The number of training cycles to run.

        Raises:
            - ValueError:
                  - If the algorithm is unknown.
        """

        # Check arguments
        if algorithm not in ["ppo", "ddpg", "td3"]:
            raise ValueError(f"Unknown RL algorithm '{algorithm}'.")

        # Create a copy of the configuration file in the temporary directory
        src_config_file_name = os.path.abspath(
            os.path.join(
                TestTraining._top_level_dir_name,
                "lexci2/test_envs/pendulum_minion/",
                f"pendulum_env_{algorithm}_config.yaml",
            )
        )
        config_file_name = os.path.abspath(
            os.path.join(
                TestTraining._tmp_dir.name,
                f"pendulum_env_{algorithm}_config.yaml",
            )
        )
        shutil.copy(src_config_file_name, config_file_name)

        # Modify the config file such that LExCI writes its output into the
        # temporary folder and performs a validation run in every cycle
        results_dir_name = os.path.abspath(
            os.path.join(TestTraining._tmp_dir.name, "lexci_results")
        )
        with open(config_file_name, "r") as f:
            config = yaml.safe_load(f)
        config["master_config"]["output_dir"] = results_dir_name
        config["master_config"]["validation_interval"] = 1
        with open(config_file_name, "w") as f:
            f.write(yaml.dump(config))

        # Inform the user about the training's timeout
        logger.info(
            f"The timeout of the {algorithm.upper()} training is set to"
            + f" {timeout} s. You can change this value by setting the"
            + " environment variable"
            + f" `LEXCI_TEST_{algorithm.upper()}_TRAINING_TIMEOUT`."
        )

        # Create the background processes
        master_proc = None
        minion_proc = None
        try:
            # Start the Universal Master
            logger.info(f"Starting the Universal {algorithm.upper()} Master...")
            cmd = f'exec /bin/bash -c "{TestTraining._venv_activation_cmd}'
            cmd += (
                f" && Lexci2Universal{algorithm.title()}Master"
                + f' {config_file_name}"'
            )
            master_proc = subprocess.Popen(
                cmd,
                shell=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Wait for the server to come online
            time.sleep(15)
            logger.info("... done.")

            # Start the Pendulum Minion
            logger.info("Starting the Pendulum Minion...")
            pendulum_minion_name = os.path.abspath(
                os.path.join(
                    TestTraining._top_level_dir_name,
                    "lexci2/test_envs/pendulum_minion/pendulum_minion.py",
                )
            )
            cmd = f'exec /bin/bash -c "{TestTraining._venv_activation_cmd}'
            cmd += f' && python3.9 {pendulum_minion_name} {algorithm}"'
            minion_proc = subprocess.Popen(
                cmd,
                shell=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Wait for the Minion to connect to the Master
            time.sleep(15)
            logger.info("... done.")

            # Names of the log file and the validation folder
            training_dir_name = find_newest_folder(results_dir_name)
            if training_dir_name is None:
                self.fail("Could not find the training output folder.")
            log_file_name = os.path.abspath(
                os.path.join(training_dir_name, "log.csv")
            )
            log_file_reader = CsvLogReader(log_file_name)
            validation_folder_name = os.path.abspath(
                os.path.join(training_dir_name, "Validation_Data")
            )

            # Wait until the required number of cycles has been completed
            logger.info(
                f"Waiting for the agent to complete {num_training_cycles}"
                + " training cycles..."
            )
            t_start = datetime.datetime.now()
            while True:
                # Wait for new data to be generated
                time.sleep(60)

                # Check if the required number of cycles has been completed
                current_cycle_no = log_file_reader.get_num_cycles()
                logger.info(f"... {current_cycle_no} runs already finished...")
                if current_cycle_no >= num_training_cycles:
                    break

                # Check whether the job has timed out
                t_now = datetime.datetime.now()
                if (t_now - t_start).total_seconds() >= timeout:
                    self.fail(f"The {algorithm.upper()} training has time out.")
            logger.info("... done.")
        except Exception as e:
            self.fail(
                f"The following exception was caught during training: {e}"
            )
        finally:
            # Stop the Master and the Minion processes
            logger.info(
                "Killing the background processes of the Master and the"
                + " Minion..."
            )
            if master_proc is not None:
                master_proc.kill()
                master_proc.wait()
                master_proc = None
            if minion_proc is not None:
                minion_proc.kill()
                minion_proc.wait()
                minion_proc = None
            logger.info("... done.")

        # Analyze the training progress
        logger.info("Analyzing the training progress...")
        ## Load and smoothen the reference data
        ref_file_name = os.path.abspath(
            os.path.join(
                TestTraining._top_level_dir_name,
                "tests/data/",
                f"pendulum_environment_{algorithm}_reference_log.csv",
            )
        )
        ref_file_reader = CsvLogReader(ref_file_name)
        ref_training_rewards = ref_file_reader.get_episode_reward_mean_list()
        ref_training_rewards = np.array(
            ref_training_rewards[:num_training_cycles], dtype=np.float32
        )
        ref_training_rewards = apply_moving_average(
            ref_training_rewards, TestTraining._MOVING_AVERAGE_KERNEL_SIZE
        )
        ## Retrieve and smoothen the training data
        training_rewards = log_file_reader.get_episode_reward_mean_list()
        training_rewards = np.array(
            training_rewards[:num_training_cycles], dtype=np.float32
        )
        training_rewards = apply_moving_average(
            training_rewards, TestTraining._MOVING_AVERAGE_KERNEL_SIZE
        )
        ## Calculate the root mean squared error and fail if it is too large
        rmse = calc_rmse(ref_training_rewards, training_rewards)
        if rmse > TestTraining._MAX_ALLOWED_RMSE:
            self.fail(
                f"The {algorithm.upper()} training has a RMSE of {rmse:.1f}"
                + " when comparing with the reference, but the maximum allowed"
                + " RMSE of the test case is"
                + f" {TestTraining._MAX_ALLOWED_RMSE:.1f}."
            )
        logger.info("... done.")

        # Analyze the validation performance
        logger.info(
            "Checking whether the agent achieved a return of"
            + f" {TestTraining._MIN_SUCCESSFUL_VALIDATION_RETURN} or better in"
            + " a validation run..."
        )
        for validation_file_name in list_files(validation_folder_name):
            with open(validation_file_name, "r") as f:
                # Compute the return
                episode_return = 0
                for row in list(csv.DictReader(f, delimiter=";")):
                    episode_return += float(row["reward"])

                # Check whether the agent performed well enough in the
                # validation
                if (
                    episode_return
                    >= TestTraining._MIN_SUCCESSFUL_VALIDATION_RETURN
                ):
                    break
        else:
            self.fail(
                "No validation run has a return which is better than"
                + f" {TestTraining._MIN_SUCCESSFUL_VALIDATION_RETURN}."
            )
        logger.info("... done.")

    def test_ppo(self) -> None:
        """Train with a PPO agent."""

        # Constants
        NUM_TRAINING_CYCLES = 751
        TRAINING_TIMEOUT = os.environ.get(
            "LEXCI_TEST_PPO_TRAINING_TIMEOUT", 21600
        )

        # Run the actual test
        self._run_training_test("ppo", TRAINING_TIMEOUT, NUM_TRAINING_CYCLES)

    def test_ddpg(self) -> None:
        """Train with a DDPG agent."""

        # Constants
        NUM_TRAINING_CYCLES = 51
        TRAINING_TIMEOUT = os.environ.get(
            "LEXCI_TEST_DDPG_TRAINING_TIMEOUT", 9000
        )

        # Run the actual test
        self._run_training_test("ddpg", TRAINING_TIMEOUT, NUM_TRAINING_CYCLES)

    def test_td3(self) -> None:
        """Train with a TD3 agent."""

        # Constants
        NUM_TRAINING_CYCLES = 51
        TRAINING_TIMEOUT = os.environ.get(
            "LEXCI_TEST_TD3_TRAINING_TIMEOUT", 9000
        )

        # Run the actual test
        self._run_training_test("td3", TRAINING_TIMEOUT, NUM_TRAINING_CYCLES)


if __name__ == "__main__":
    unittest.main()
