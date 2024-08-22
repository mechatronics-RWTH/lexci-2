"""TODO.

File:   lexci2/universal_masters/config_creator.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2024-08-22


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

import sys
import argparse
import copy
import json
import logging

from typing import Any


# Create the logger
logger = logging.getLogger(__name__)


class ConfigCreator:
    """Helper for creating configuration files for LExCI's Universal Masters."""

    def __init__(
        self,
        master_config: dict[str, Any],
        algo_config: dict[str, Any],
        algo_name: str,
    ) -> None:
        """Initialize the config creator.

        Arguments:
            - master_config: dict[str, Any]
                  Dictionary containing general settings for the Master.
            - algo_config: dict[str, Any]
                  Dictionary containing algorithm-related settings.
            - algo_name: str
                  Name of the RL algorithm that is used.
        """

        self._master_config = copy.deepcopy(master_config)
        self._algo_config = copy.deepcopy(algo_config)
        self._algo_name = algo_name

    def write_config_file(self, output_file_name: str) -> None:
        """Save the configuration as a JSON-file.

        Arguments:
            - output_file_name: str
                  Name of the JSON-file.
        """

        # Remove all keys from the algorithm's dictionary that are not
        # JSON-serializable
        keys_to_remove = []
        for k, v in self._algo_config.items():
            if v is not None and type(v) not in [
                dict,
                list,
                str,
                int,
                float,
                bool,
            ]:
                logger.info(
                    f"Removing key '{k}' with value '{v}' from the algorithm's"
                    + " configuration as it isn't JSON-serializable."
                )
                keys_to_remove.append(k)
        for k in keys_to_remove:
            del self._algo_config[k]

        # Write the JSON file
        config = {
            "master_config": self._master_config,
            self._algo_name: self._algo_config,
        }
        with open(output_file_name, "w") as f:
            json.dump(config, f, indent=2)

    def run(self) -> None:
        """Read the output file name from the command-line arguments and write
        the configuration into a JSON-file.
        """

        # Parse command line arguments
        arg_parser = argparse.ArgumentParser(
            description=(
                "Save LExCI's Universal Master configuration as a JSON-file."
            )
        )
        arg_parser.add_argument(
            "output_file", type=str, help="Output file to write."
        )
        cli_args = arg_parser.parse_args(sys.argv[1:])

        self.write_config_file(cli_args.output_file)
