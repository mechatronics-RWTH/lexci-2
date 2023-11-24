"""Postprocessing script for ControlDesk recorder CSVs.

File:   controldesk_csv_postprocessor.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2022-10-04


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


from lexci2.minion.controllers.controldesk_controller import (
    ControlDeskController,
)
from lexci2.utils.csv_export import export_episode_csv

import sys
import argparse
import csv


# Parse command line arguments
arg_parser = argparse.ArgumentParser(
    description=("Postprocessing script for" " ControlDesk recorder CSVs.")
)
arg_parser.add_argument(
    "controldesk_csv", type=str, help=("ControlDesk" " recorder CSV file.")
)
arg_parser.add_argument(
    "output_csv", type=str, help=("Postprocessed output CSV" " file.")
)
cli_args = arg_parser.parse_args(sys.argv[1:])


# Read the ControlDesk CSV and postprocess it
cdc = ControlDeskController()
eps = cdc.extract_csv_data(cli_args.controldesk_csv, False)
if len(eps) != 1:
    raise RuntimeError("Didn't find the correct number of episodes.")
export_episode_csv(eps[0], cli_args.output_csv)
