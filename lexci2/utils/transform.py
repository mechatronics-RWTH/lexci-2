"""Transformations to and from observation/action spaces.

File:   lexci2/utils/transform.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2022-04-27


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


import copy
import numpy as np


def transform_linear(
    x: np.ndarray,
    x_min: np.ndarray,
    x_max: np.ndarray,
    y_min: np.ndarray,
    y_max: np.ndarray,
) -> np.ndarray:
    """Transform data linearly between two closed intervals.

    Arguments:
        - x: np.ndarray
              Input data in [x_min, x_max].
        - x_min: np.ndarray
              Lower bound of the input.
        - x_max: np.ndarray
              Upper bound of the input.
        - y_min: np.ndarray
              Lower bound of the output.
        - y_max: np.ndarray
              Upper bound of the output.

    Returns:
        - y: np.ndarray
              Transformed data in [y_min, y_max].
    """

    y = copy.deepcopy(x)  # in [x_min, x_max]
    y = (y - x_min) / (x_max - x_min)  # in [0, 1]
    y = (y_max - y_min) * y + y_min  # in [y_min, y_max]
    return y


def transform_tanh(
    x: np.ndarray, y_min: np.ndarray, y_max: np.ndarray
) -> np.ndarray:
    """Transform from real values to a closed interval using a hyperbolic
    tangent.

    Arguments:
        - x: np.ndarray
              Input data in [-inf, inf].
        - y_min: np.ndarray
              Lower bound of the output.
        - y_max: np.ndarray
              Upper bound of the output.

    Returns:
        - y: np.ndarray
              Transformed data in [y_min, y_max].
    """

    y = copy.deepcopy(x)  # in [-inf, inf]
    y = (np.tanh(y) + 1) / 2  # in [0, 1]
    y = (y_max - y_min) * y + y_min  # in [y_min, y_max]
    return y


def transform_atanh(
    x: np.ndarray, x_min: np.ndarray, x_max: np.ndarray
) -> np.ndarray:
    """Transform values in a closed interval to real space using an inverse
    hyperbolic tangent.

    Arguments:
        - x: np.ndarray
              Input data in [x_min, x_max].
        - x_min: np.ndarray
              Lower bound of the input.
        - x_max: np.ndarray
              Upper bound of the input.

    Returns:
        - y: np.ndarray
              Transformed data in real space.
    """

    y = copy.deepcopy(x)  # in [x_min, x_max]
    y = 2 * ((y - x_min) / (x_max - x_min)) - 1  # in [-1, 1]
    y = np.arctanh(y)  # in [-inf, inf]
    return y
