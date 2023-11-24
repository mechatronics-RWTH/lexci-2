"""Script for converting an ONNX model to the byte format used by LExCI 2's
Reinforcement Learning Block.

File:   onnx2rlblock.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2023-04-13


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


import tensorflow as tf
from tensorflow import keras
import onnx
import onnx_tf
import uuid
import os
import sys
import argparse
import shutil


# Parse command line arguments
arg_parser = argparse.ArgumentParser(
    description=(
        "Script for converting an ONNX model to the byte format used by"
        + " LExCI 2's Reinforcement Learning Block."
    )
)
arg_parser.add_argument(
    "--padding_size",
    type=int,
    default=None,
    help=("Pad" + " the bytes of the neural network to this size using zeros."),
)
arg_parser.add_argument("onnx_file", type=str, help="The ONNX file to convert.")
arg_parser.add_argument(
    "rlblock_bytes_file",
    type=str,
    help=(
        "Text file that will contain the bytes of the neural network for"
        + " LExCI 2's RL block."
    ),
)
cli_args = arg_parser.parse_args(sys.argv[1:])


# Import and prepare the model
onnx_model = onnx.load(cli_args.onnx_file)
tf_model = onnx_tf.backend.prepare(onnx_model)


# Convert the model to TensorFlow Lite's format and extract its bytes
while True:
    temp_folder_name = os.path.join(os.getcwd(), str(uuid.uuid4()))
    if not os.path.exists(temp_folder_name):
        break
os.mkdir(temp_folder_name)
tf_model.export_graph(temp_folder_name)
tflite_converter = tf.lite.TFLiteConverter.from_saved_model(temp_folder_name)
tflite_converter.experimental_new_converter = False
tflite_converter.experimental_quantizer = False
tflite_converter.experimental_enable_resource_variables = False
nn_bytes = tflite_converter.convert()
if os.path.exists(temp_folder_name):
    shutil.rmtree(temp_folder_name)


# Pad the bytes if required
if cli_args.padding_size is not None:
    nn_size = len(nn_bytes)
    if nn_size < cli_args.padding_size:
        nn_bytes.extend((cli_args.padding_size - nn_size) * [0x00])


# Write the bytes to a the text file
with open(cli_args.rlblock_bytes_file, "w") as f:
    f.write(str(list(nn_bytes)))
