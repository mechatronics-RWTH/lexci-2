"""Script for printing the weights of neural network.

File:   PrintNnH5.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2023-01-12


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


# Parse command line arguments
import argparse
import sys

arg_parser = argparse.ArgumentParser(
    description=("Script for printing the" " weights of a neural network.")
)
arg_parser.add_argument(
    "keras_h5_file",
    type=str,
    help=("A Keras .h5 file" " containing the neural network."),
)
arg_parser.add_argument(
    "--output-file",
    type=str,
    help=("Optional output file" " to write the weights to."),
    default=None,
)
cli_args = arg_parser.parse_args(sys.argv[1:])


import tensorflow as tf


# Load the neural network, extract the weights and print them
keras_model = tf.keras.models.load_model(cli_args.keras_h5_file)
weights = str(keras_model.get_weights())
print(weights)

# Store the weights in a text file if the user so wishes
if cli_args.output_file is not None:
    with open(cli_args.output_file, "w") as f:
        f.write(weights)
