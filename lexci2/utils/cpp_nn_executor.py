"""Neural network executor that runs the model using C++ code.

File:   lexci2/utils/cpp_nn_executor.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2023-02-24


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


import ctypes
import os
import numpy as np


# Load the 'libnnexec' library and tell Python about the parameter and return
# types of its functions
libnnexec = ctypes.CDLL(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "libnnexec.so")
)

libnnexec.createNeuralNetwork.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte),
    ctypes.c_size_t,
    ctypes.c_size_t,
]
libnnexec.createNeuralNetwork.restype = ctypes.c_void_p

libnnexec.destroyNeuralNetwork.argtypes = [ctypes.c_void_p]
libnnexec.destroyNeuralNetwork.restype = None  # = void

libnnexec.getNumInputLayers.argtypes = [ctypes.c_void_p]
libnnexec.getNumInputLayers.restype = ctypes.c_size_t

libnnexec.getInputLayerSize.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libnnexec.getInputLayerSize.restype = ctypes.c_size_t

libnnexec.getNumOutputLayers.argtypes = [ctypes.c_void_p]
libnnexec.getNumOutputLayers.restype = ctypes.c_size_t

libnnexec.getOutputLayerSize.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libnnexec.getOutputLayerSize.restype = ctypes.c_size_t

libnnexec.setInputData.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
]
libnnexec.setInputData.restype = None  # = void

libnnexec.execute.argtypes = [ctypes.c_void_p]
libnnexec.execute.restype = None  # = void

libnnexec.getOutputData.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
]
libnnexec.getOutputData.restype = None  # = void


class CppNnExecutor:
    """Class for executing a neural network by running the model through C++
    code."""

    def __init__(
        self, nn_data: bytes, tensor_arena_size: int = 1000000
    ) -> None:
        """Initialize the neural network.

        Arguments:
            - nn_data: bytes
                  Bytes of the neural network in the TensorFlow Lite format.
            - tensor_arena_size: int (Default: 1000000, Unit: B)
                  Size of the tensor arena.
        """

        # Convert the NN data
        conv_nn_data = (len(nn_data) * ctypes.c_ubyte)()
        for i in range(len(nn_data)):
            conv_nn_data[i] = nn_data[i]
        conv_nn_data_len = ctypes.c_size_t(len(conv_nn_data))

        # Convert the tensor arena size
        conv_tensor_arena_size = ctypes.c_size_t(tensor_arena_size)

        # Create the C++ neural network object
        self._ptr_nn = libnnexec.createNeuralNetwork(
            conv_nn_data, conv_nn_data_len, conv_tensor_arena_size
        )

        # Get the dimensions of the NN's input layers
        self._num_input_layers = libnnexec.getNumInputLayers(self._ptr_nn)
        self._input_layer_sizes = []
        for i in range(self._num_input_layers):
            layer_size = libnnexec.getInputLayerSize(self._ptr_nn, i)
            self._input_layer_sizes.append(layer_size)

        # Get the dimensions of the NN's output layers
        self._num_output_layers = libnnexec.getNumOutputLayers(self._ptr_nn)
        self._output_layer_sizes = []
        for i in range(self._num_output_layers):
            layer_size = libnnexec.getOutputLayerSize(self._ptr_nn, i)
            self._output_layer_sizes.append(layer_size)

    def __del__(self) -> None:
        """Last resort attempt to destroy the object."""

        if self._ptr_nn is not None:
            libnnexec.destroyNeuralNetwork(ctypes.c_void_p(self._ptr_nn))
            self._ptr_nn = None

    def get_num_input_layers(self) -> int:
        """Get the number of input layers in the neural network.

        Returns:
            - _: int
                  The number of input layers.
        """

        return self._num_input_layers

    def get_input_layer_size(self, idx: int) -> int:
        """Get the size of an input layer.

        Arguments:
            - idx: int
                  Index of an input layer in the neural network.

        Returns:
            - _: int
                  Size of the input layer.
        """

        return self._input_layer_sizes[idx]

    def get_num_output_layers(self) -> int:
        """Get the number of output layers in the neural network.

        Returns:
            - _: int
                  The number of output layers.
        """

        return self._num_output_layers

    def get_output_layer_size(self, idx: int) -> int:
        """Get the size of an output layer.

        Arguments:
            - idx: int
                  Index of an output layer in the NN.

        Returns:
            - _: int
                  Size of the output layer.
        """

        return self._output_layer_sizes[idx]

    def predict(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        """Execute the neural network with a given input.

        Arguments:
            - inputs: list[np.ndarray]
                  A list that contains a NumPy array with input data for each
                  input layer.

        Returns:
            - _: list[np.ndarray]
                  A list that contains a NumPy array with output data for each
                  output layer.
        """

        # Set the input data
        for i in range(self._num_input_layers):
            layer_size = self._input_layer_sizes[i]
            layer_input = (layer_size * ctypes.c_float)(*(inputs[i][0]))
            libnnexec.setInputData(self._ptr_nn, i, layer_input, layer_size)

        # Run the model
        libnnexec.execute(self._ptr_nn)

        # Get the output data
        outputs = []
        for i in range(self._num_output_layers):
            layer_size = self._output_layer_sizes[i]
            layer_output = (layer_size * ctypes.c_float)()
            libnnexec.getOutputData(self._ptr_nn, i, layer_output, layer_size)
            outputs.append(np.array(layer_output, dtype=np.float32))

        return outputs
