"""
Script for converting a neural network in Keras's .h5 format into a list of
decimal integers representing the bytes of a TensorFlow Lite model.

File:   KerasH5ToTfLiteBytes.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
Date:   2022-07-26


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
    description="Script for converting a"
    + " neural network in Keras's .h5 format into a list of decimal integers"
    + " representing the bytes of a TensorFlow Lite model."
)
arg_parser.add_argument(
    "--padding_size",
    "-p",
    type=int,
    dest="padding_size",
    help="Number of bytes to pad the output to using zeros (default: no"
    + " padding).",
    default=None,
)
arg_parser.add_argument("keras_h5_file", type=str, help="A Keras .h5 file.")
arg_parser.add_argument(
    "output_file", type=str, help="Output file containing" " the TF Lite bytes."
)
parsed_args = arg_parser.parse_args(sys.argv[1:])


import tensorflow as tf


# Convert
keras_model = tf.keras.models.load_model(parsed_args.keras_h5_file)
tflite_converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_converter.experimental_new_converter = False
tflite_converter.experimental_quantizer = False
tflite_converter.experimental_enable_resource_variables = False
data = list(tflite_converter.convert())


# Pad the data if the user so wishes
if parsed_args.padding_size is not None:
    data_size = len(data)
    if data_size < parsed_args.padding_size:
        data.extend((parsed_args.padding_size - data_size) * [0])


# Write the output file
with open(parsed_args.output_file, "w") as f:
    f.write(str(data))
