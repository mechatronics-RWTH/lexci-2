"""A wrapper for the LExCI Minion that restarts it everytime the system runs low
on memory.

File:   tools/minion_runner.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2024-08-16


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

import argparse
import sys
import os
import subprocess
import psutil
import time
import logging


# Logger
logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
)
logger = logging.getLogger()


def main() -> None:
    """Main function of the Minion runner."""

    # Command line arguments
    arg_parser = argparse.ArgumentParser(
        description=(
            "A wrapper for the LExCI Minion that restarts the latter whenever"
            + " it consumes to much memory."
        )
    )
    arg_parser.add_argument(
        "virtenv_activation_cmd",
        type=str,
        help="Command for activating LExCI's virtual environment.",
    )
    arg_parser.add_argument(
        "minion_file", type=str, help="Path to the Minion's Python script."
    )
    arg_parser.add_argument(
        "memory_threshold_percent",
        type=float,
        help=(
            "The memory threshold (in percent of the system's total memory)"
            + " beyond which the Minion is restarted."
        ),
    )
    cli_args = arg_parser.parse_args(sys.argv[1:])

    while True:
        proc = None
        try:
            # Run the Minion
            cmd = f'exec /bin/bash -c "{cli_args.virtenv_activation_cmd}'
            cmd += f' && python3.9 {cli_args.minion_file}"'
            proc = subprocess.Popen(cmd, shell=True)
            while True:
                memory_usage = psutil.virtual_memory().percent
                threshold = cli_args.memory_threshold_percent
                if memory_usage > threshold:
                    logger.info(
                        "The system exceeded the memory threshold"
                        + f" ({memory_usage:.1f}%/{threshold:.1f}%)."
                        + " Restarting the Minion."
                    )
                    proc.kill()
                    proc = None
                    break
                time.sleep(15)
        except KeyboardInterrupt:
            logger.info("The user pressed [CTRL]+[C]. Quitting.")
            if proc is not None:
                proc.kill()
                proc = None
            break


if __name__ == "__main__":
    main()
