"""Class for TensorFlow Light Micro's neural networks.

File:   lexci2/neural_network_modules/neural_networks/tflite_neural_network.py
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


from lexci2.neural_network_modules.neural_networks.neural_network import (
    NeuralNetwork,
)
from lexci2.utils.cpp_nn_executor import CppNnExecutor

import numpy as np


class TfliteNeuralNetwork(NeuralNetwork):
    """A TensorFlow Lite Micro neural network."""

    def __init__(
        self, nn_data: bytes, tensor_arena_size: int = 1000000
    ) -> None:
        """Initialize the neural network.

        Arguments:
            - nn_data: bytes
                  Bytes of the neural network.
            - tensor_arena_size: int (Default: 1000000, Unit: B)
                  Size of the tensor arena.
        """

        super().__init__(nn_data)
        self._nn = CppNnExecutor(nn_data, tensor_arena_size)

    def get_num_input_layers(self) -> int:
        """Get the number of input layers in the neural network.

        Returns:
            - _: int
                  The number of input layers.
        """

        return self._nn.get_num_input_layers()

    def get_input_layer_size(self, idx: int) -> int:
        """Get the size of an input layer.

        Arguments:
            - idx: int
                  Index of an input layer.

        Returns:
            - _: int
                  Size of the input layer.
        """

        return self._nn.get_input_layer_size(idx)

    def get_num_output_layers(self) -> int:
        """Get the number of output layers.

        Returns:
            - _: int
                  The number of output layers.
        """

        return self._nn.get_num_output_layers()

    def get_output_layer_size(self, idx: int) -> int:
        """Get the size of an output layer.

        Arguments:
            - idx: int
                  Index of an output layer.

        Returns:
            - _: int
                  Size of the output layer.
        """

        return self._nn.get_output_layer_size(idx)

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

        return self._nn.predict(inputs)
