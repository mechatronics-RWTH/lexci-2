"""Helper functions for working with TensorFlow.

File:   lexci2/utils/tf_helper.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2024-11-18


Copyright 2024 Teaching and Research Area Mechatronics in Mobile Propulsion,
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

import logging
from typing import Optional


# Create the logger
logger = logging.getLogger(__name__)


def keras_h5_file_to_tf_bytes(
    h5_file_name: str, padding_size: Optional[int] = None
) -> bytes:
    """Import a Keras model from an h5-file, convert it to TensorFlow Lite
    Micro, and return its bytes.

    Arguments:
        - h5_file_name: str
              The h5-file to import.
        - padding_size: Optional[int] (Default: None)
              If not `None`, this defines the number of bytes to pad the output
              to with zeros.

    Returns:
        - _: bytes
              The bytes of the imported and converted TensorFlow Lite Micro
              model.
    """

    # Load and convert the model
    keras_model = tf.keras.models.load_model(h5_file_name)
    tflite_converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_converter.experimental_new_converter = False
    tflite_converter.experimental_quantizer = False
    tflite_converter.experimental_enable_resource_variables = False
    data = tflite_converter.convert()

    # Pad the data if required
    if padding_size is not None:
        data_size = len(data)
        if data_size < padding_size:
            data = data + (padding_size - data_size) * b"\x00"
        else:
            logger.info(
                f"Not padding the model data to {padding_size} B as the model"
                + f" is already {data_size} B big."
            )

    return data
