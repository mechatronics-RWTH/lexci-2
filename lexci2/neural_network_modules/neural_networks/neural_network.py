"""Abstract base class for neural networks.

File:   lexci2/neural_network_modules/neural_networks/neural_network.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2023-03-02


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


from abc import ABCMeta, abstractmethod
import numpy as np


class NeuralNetwork(metaclass=ABCMeta):
    """Abstract base class for neural networks."""

    def __init__(self, nn_data: bytes) -> None:
        """Initialize the neural network.

        Arguments:
            - nn_data: bytes
                  Bytes of the neural network.
        """

        self._nn_data = nn_data
        self._nn = None  # To be set by sub-classes

    @abstractmethod
    def get_num_input_layers(self) -> int:
        """Get the number of input layers in the neural network.

        Returns:
            - _: int
                  The number of input layers.
        """

        raise NotImplementedError

    @abstractmethod
    def get_input_layer_size(self, idx: int) -> int:
        """Get the size of an input layer.

        Arguments:
            - idx: int
                  Index of an input layer.

        Returns:
            - _: int
                  Size of the input layer.
        """

        raise NotImplementedError

    @abstractmethod
    def get_num_output_layers(self) -> int:
        """Get the number of output layers.

        Returns:
            - _: int
                  The number of output layers.
        """

        raise NotImplementedError

    @abstractmethod
    def get_output_layer_size(self, idx: int) -> int:
        """Get the size of an output layer.

        Arguments:
            - idx: int
                  Index of an output layer.

        Returns:
            - _: int
                  Size of the output layer.
        """

        raise NotImplementedError

    @abstractmethod
    def predict(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        """Execute the neural network with a given input.

        Arguments:
            - inputs: list[np.ndarray]:
                  A list that contains a NumPy array with input data for each
                  input layer.

        Returns:
            - _: list[np.ndarray]
                  A list that contains a NumPy array with output data for each
                  output layer.
        """

        raise NotImplementedError
