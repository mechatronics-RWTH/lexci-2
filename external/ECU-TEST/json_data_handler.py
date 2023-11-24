"""Functions that allow TraceTronic's ECU-TEST to read and write JSON files.

These are used inside the read and write package to exchange data between
ECU-TEST and LExCI.

File:   external/ECU-TEST/json_data_handler.py
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


from tts.core.package.dataStructures.Vector import FullVector

import json


def read_json_data(data_file):
    """Read a JSON file and use its contents to update a dictionary.

    Arguments:
      - data_file: str
          Name of the JSON file to read.

    Returns:
      - _: dict[str, Any]
          Dictionary with the variable names and values that have been read.
    """

    d = {}
    with open(data_file, "r") as f:
        d = json.load(f)
    return d


def write_json_data(data_file, variables):
    """Write a dictionary containing ECU-TEST variables to a JSON file.

    Arguments:
      - data_file: str
          Name of the JSON file to write to.
      - variables: dict[str, Any]
          Variables to write.
    """

    # Convert variables to Pyhon data types
    d = {}
    for k, v in variables.items():
        if isinstance(v, FullVector):
            d[k] = [e[1] for e in list(v)]
        else:
            d[k] = v

    # Write the JSON file
    with open(data_file, "w") as f:
        json.dump(d, f)
