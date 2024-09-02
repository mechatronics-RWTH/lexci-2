"""A collection of mathematical helper functions.

File:   lexci2/utils/math.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2023-09-12


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
import logging


# Create the logger
logger = logging.getLogger(__name__)


def softmax(x: np.ndarray) -> np.ndarray:
    """Apply the softmax function to an array.

    Arguments:
        - x: np.ndarray
              A NumPy array.

    Returns:
        - _: np.ndarray
              An array of probabilities.
    """

    return np.exp(x) / np.sum(np.exp(x))


def inv_softmax(x: np.ndarray) -> np.ndarray:
    """Apply the inverse softmax function to an array.

    The softmax function is defined as:
        S_i = exp(x_i) / (exp(x_1) + exp(x_2) + ... + exp(x_n))
    Rearranging the equation and applying the natural logarithm yields:
        x_i = ln(S_i) + ln(exp(x_1) + exp(x_2) + ... + exp(x_n))
    The second term on the right-hand side of the equation is constant and can
    be freely chosen, i.e. the solution is not unique. This method assumes that
    the constant is 0.

    Arguments:
        - x: np.ndarray
              A NumPy array of probabilities.

    Returns:
        - _: np.ndarray
              The array after the inverse softmax function has been applied.
    """

    return np.log(x)


def apply_moving_average(x: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply a moving average filter to an input array.

    Arguments:
        - x: np.ndarray
              The input data which will not be modified.
        - kernel_size: int
              The size of the moving average kernel. If it is even, the number
              will be rounded up to the next odd value.

    Returns:
        - _: np.ndarray
              A new array containing the smoothed data.
    """

    # Check and adjust the kernel size if necessary
    if kernel_size % 2 != 1:
        logger.info(
            f"Using a kernel size of {kernel_size + 1} instead {kernel_size}"
            + " which was passed to the function."
        )
        kernel_size += 1

    # Perform the smoothing
    x_smooth = np.zeros(x.shape, dtype=x.dtype)
    kernel_offset_limit = int(kernel_size / 2)
    for i in range(len(x)):
        for j in range(-kernel_offset_limit, kernel_offset_limit + 1, 1):
            if i + j < 0:
                x_smooth[i] += x[0]
            elif i + j >= len(x):
                x_smooth[i] += x[-1]
            else:
                x_smooth[i] += x[i + j]
        x_smooth[i] /= kernel_size

    return x_smooth


def calc_rmse(x_1: np.ndarray, x_2: np.ndarray) -> float:
    """Calculate the root mean squared error (RMSE) of two datasets.

    Arguments:
        - x_1: np.ndarray
              The first dataset which is considered the reference.
        - x_2: np.ndarray
              The second dataset.

    Returns:
        - _: float
              The datasets' RMSE. A small value indicates a close resemblance
              between the two.

    Raises:
        - ValueError:
              - If the two datasets aren't of equal size.
    """

    # Check the datasets
    if len(x_1) != len(x_2):
        raise ValueError("The datasets aren't of equal size.")

    # Compute the RMSE
    rmse = 0.0
    for i in range(len(x_1)):
        rmse += (x_2[i] - x_1[i]) ** 2
    rmse /= len(x_1)
    rmse = np.sqrt(rmse)

    return rmse
