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


import numpy as np


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
