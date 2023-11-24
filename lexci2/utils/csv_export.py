"""Tools for exporting CSVs.

File:   utils/csv_export.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2022-05-24


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


from lexci2.data_containers import Experience, Episode, Cycle

import csv
import numpy as np
from typing import Union


def export_episode_csv(eps: Episode, file_name: str) -> None:
    """Export an episode to a CSV file.

    This function assumes that all experiences are identical in terms of
    observation size, action size, auxiliary data fields, etc.

    Arguments:
        - eps: Episode
              A LExCI episode.
        - file_name: str
              Name of the CSV file.

    Raises:
        - ValueError
    """

    if len(eps.exps) == 0:
        raise ValueError("The episode contains no experiences.")

    with open(file_name, "w") as f:
        # Write CSV header
        csv_writer = csv.writer(f, delimiter=";", quotechar='"')
        csv_writer.writerow(_get_csv_header_list(eps.exps[0]))

        # Write data
        for exp in eps.exps:
            row = []

            # Standard entries
            row.extend(exp.obs.tolist())
            row.extend(exp.action.tolist())
            row.extend(exp.new_obs.tolist())
            row.append(exp.reward)
            row.append(exp.done)
            if exp.t is not None:
                row.append(exp.t)

            # Auxiliary data fields
            for k in sorted(exp.aux_data.keys()):
                v = exp.aux_data[k]
                if isinstance(v, list):
                    row.extend(v)
                elif isinstance(v, np.ndarray):
                    row.extend(v.tolist())
                else:
                    row.append(v)

            csv_writer.writerow(row)


def _get_csv_header_list(exp: Experience) -> list[str]:
    """Determine the row names in the CSV header.

    CSV headers may differ from environment to environment, e.g. because the
    observation space is different or there are different auxiliary data fields.
    This function makes sure that all lists and NumPy arrays are split into
    their individual entries by adding an index to the CSV row name.

    Arguments:
        - eps: Episode
              A LExCI episode.

    Returns:
        - _: list[str]
              List containing the names of the CSV columns of the exported
              episode.
    """

    row_names = []

    # Standard fields
    row_names.extend(_get_row_names(exp.obs, "obs"))
    row_names.extend(_get_row_names(exp.action, "action"))
    row_names.extend(_get_row_names(exp.new_obs, "new_obs"))
    row_names.append("reward")
    row_names.append("done")
    if exp.t is not None:
        row_names.append("t")

    # Auxiliary data fields
    for k in sorted(exp.aux_data.keys()):
        v = exp.aux_data[k]
        if isinstance(v, list) or isinstance(v, np.ndarray):
            row_names.extend(_get_row_names(v, k))
        else:
            row_names.append(k)

    return row_names


def _get_row_names(l: Union[list, np.ndarray], name: str) -> list[str]:
    """Generate a list of indexed entry names of a list or NumPy array based on
    its size and name.

    Arguments:
        - l: Union[list, np.ndarray]
              List or NumPy array.
        - name: str
              Name of the list or NumPy array.

    Returns:
        - _: list[str]
              Indexed entry names.
    """

    row_names = []
    for i in range(len(l)):
        row_names.append(f"{name}[{i}]")
    return row_names
