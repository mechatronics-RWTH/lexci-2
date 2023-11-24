"""Class for Keras' neural networks.

File:   lexci2/neural_network_modules/neural_networks/keras_neural_network.py
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

import uuid
import os
import numpy as np
import tensorflow as tf


class KerasNeuralNetwork(NeuralNetwork):
    """A Keras neural network."""

    def __init__(self, nn_data: bytes) -> None:
        """Initialize the neural network.

        Arguments:
            - nn_data: bytes
                  Bytes of the neural network.
        """

        super().__init__(nn_data)

        # Write the bytes into a temporary file
        while True:
            temp_file_name = f"{uuid.uuid4()}.h5"
            if not os.path.exists(temp_file_name):
                break
        with open(temp_file_name, "wb") as temp_file:
            temp_file.write(nn_data)

        # Import the model
        self._nn = tf.keras.models.load_model(temp_file_name)

        # Remove the temporary file
        if os.path.exists(temp_file_name):
            os.remove(temp_file_name)

    def get_num_input_layers(self) -> int:
        """Get the number of input layers in the neural network.

        Returns:
            - _: int
                  The number of input layers.
        """

        raise NotImplementedError

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

    def get_num_output_layers(self) -> int:
        """Get the number of output layers.

        Returns:
            - _: int
                  The number of output layers.
        """

        raise NotImplementedError

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

        result = self._nn(inputs, training=False)

        if type(result) is list:
            return [e.numpy() for e in result]
        else:
            return result.numpy()
