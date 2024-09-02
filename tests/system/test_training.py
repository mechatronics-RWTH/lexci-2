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

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove the temporary folder."""

        if cls._tmp_dir is not None:
            logger.info("Removing the temporary directory...")
            cls._tmp_dir.cleanup()
            cls._tmp_dir = None
            logger.info("... done.")

    def test_ppo(self) -> None:
        """Train with a PPO agent."""

        # Constants
        NUM_TRAINING_CYCLES = 751
        MOVING_AVERAGE_KERNEL_SIZE = 11
        MAX_ALLOWED_RMSE = 200
        MIN_SUCCESSFUL_VALIDATION_RETURN = -400

        # Create a copy of the configuration file in the temporary directory
        src_config_file_name = os.path.abspath(
            os.path.join(
                TestTraining._top_level_dir_name,
                "lexci2/test_envs/pendulum_minion/pendulum_env_ppo_config.yaml",
            )
        )
        config_file_name = os.path.abspath(
            os.path.join(
                TestTraining._tmp_dir.name, "pendulum_env_ppo_config.yaml"
            )
        )
        shutil.copy(src_config_file_name, config_file_name)

        # Modify the config file such that LExCI writes its output into the
        # temporary folder
        results_dir_name = os.path.abspath(
            os.path.join(TestTraining._tmp_dir.name, "lexci_results")
        )
        with open(config_file_name, "r") as f:
            config = yaml.safe_load(f)
        config["master_config"]["output_dir"] = results_dir_name
        with open(config_file_name, "w") as f:
            f.write(yaml.dump(config))

        # Get the value of the training timeout
        PPO_TRAINING_TIMEOUT = os.environ.get(
            "LEXCI_TEST_PPO_TRAINING_TIMEOUT", 7200
        )
        logger.info(
            f"The timeout of the PPO training is set to {PPO_TRAINING_TIMEOUT}"
            + " s. You can change this value by setting the environment"
            + " variable `LEXCI_TEST_PPO_TRAINING_TIMEOUT`."
        )

        # Start the Universal Master
        logger.info("Starting the Universal PPO Master...")
        config_file_name = os.path.abspath(
            os.path.join(
                TestTraining._top_level_dir_name,
                "lexci2/test_envs/pendulum_minion/pendulum_env_ppo_config.yaml",
            )
        )
        cmd = f'exec /bin/bash -c "{TestTraining._venv_activation_cmd}'
        cmd += f' && Lexci2UniversalPpoMaster {config_file_name}"'
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
        cmd += f' && python3.9 {pendulum_minion_name} ppo"'
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
            f"Waiting for the agent to complete {NUM_TRAINING_CYCLES} training"
            + " cycles..."
        )
        t_start = datetime.datetime.now()
        while True:
            # Wait for new data to be generated
            time.sleep(60)

            # Check if the required number of cycles has been completed
            current_cycle_no = log_file_reader.get_num_cycles()
            logger.info(f"... {current_cycle_no} runs already finished...")
            if current_cycle_no >= NUM_TRAINING_CYCLES:
                break

            # Check whether the job has timed out
            t_now = datetime.datetime.now()
            if (t_now - t_start).total_seconds() >= PPO_TRAINING_TIMEOUT:
                self.fail("The PPO training has time out.")
        logger.info("... done.")

        # Analyze the training progress
        logger.info("Analyzing the training progress...")
        ## Load and smoothen the reference data
        ref_file_name = os.path.abspath(
            os.path.join(
                TestTraining._top_level_dir_name,
                "tests/data/pendulum_environment_ppo_reference_log.csv",
            )
        )
        ref_file_reader = CsvLogReader(ref_file_name)
        ref_training_rewards = ref_file_reader.get_episode_reward_mean_list()[
            :NUM_TRAINING_CYCLES
        ]
        ref_training_rewards = apply_moving_average(
            ref_training_rewards, MOVING_AVERAGE_KERNEL_SIZE
        )
        ## Retrieve and smoothen the training data
        training_rewards = log_file_reader.get_episode_reward_mean_list()[
            :NUM_TRAINING_CYCLES
        ]
        training_rewards = apply_moving_average(
            training_rewards, MOVING_AVERAGE_KERNEL_SIZE
        )
        ## Calculate the root mean squared error and fail if it is too large
        rmse = calc_rmse(ref_training_rewards, training_rewards)
        if rmse > MAX_ALLOWED_RMSE:
            self.fail(
                f"The PPO training has a RMSE of {rmse:.1f} when comparing with"
                + " the reference, but the maximum allowed RMSE of the test"
                + f" case is {MAX_ALLOWED_RMSE:.1f}."
            )
        logger.info("... done.")

        # Analyze the validation performance
        logger.info(
            "Checking whether the agent achieved a return of"
            + f" {MIN_SUCCESSFUL_VALIDATION_RETURN} or better in a validation"
            + " run..."
        )
        for validation_file_name in list_files(validation_folder_name):
            with open(validation_file_name, "r") as f:
                # Compute the return
                episode_return = 0
                for row in list(csv.DictReader(f, delimiter=";")):
                    episode_return += float(row["reward"])

                # Check whether the agent performed well enough in the
                # validation
                if episode_return >= MIN_SUCCESSFUL_VALIDATION_RETURN:
                    break
        else:
            self.fail(
                "No validation run has a return which is better than"
                + f" {MIN_SUCCESSFUL_VALIDATION_RETURN}. Because theys aren't"
                + " performed in every cycle, it might be that the agent had"
                + " performed well between validations. However, it's very"
                + " unlikely that this happens in a successful training."
            )
        logger.info("... done.")

        # Stop the Master and the Minion processes
        logger.info(
            "Killing the background processes of the Master and the Minion..."
        )
        master_proc.kill()
        minion_proc.kill()
        logger.info("... done.")

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
