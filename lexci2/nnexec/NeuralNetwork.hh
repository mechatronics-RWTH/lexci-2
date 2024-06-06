/**
 * Helper class for creating and executing TensorFlow Lite Micro neural
 * networks.
 *
 * @file:   NeuralNetwork.hh
 * @author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
 *          Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
 *          RWTH Aachen University
 * @date:   2023-02-23
 *
 *
 * Copyright 2023 Teaching and Research Area Mechatronics in Mobile Propulsion,
 *                RWTH Aachen University
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at: http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#pragma once

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

#include <cstdint>
#include <memory>

/**
 * Helper class for creating and executing TensorFlow Lite Micro neural
 * networks.
 *
 * @class NeuralNetwork
 */
class NeuralNetwork
{
private:
  std::unique_ptr<uint8_t> sptr_nnData;
  std::unique_ptr<tflite::MicroErrorReporter> sptr_microErrorReporter;
  std::unique_ptr<tflite::AllOpsResolver> sptr_resolver;
  std::unique_ptr<uint8_t> sptr_tensorArena;
  std::unique_ptr<tflite::MicroInterpreter> sptr_interpreter;

public:
  /**
   * Constructor.
   *
   * @param[in] ptr_nnData Bytes representation of the neural network
   * @param[in] nnDataLen Size of the neural network bytes [1]
   * @param[in] tensorArenaSize Size of the tensor arena [B]. Default: 1000000 B
   */
  NeuralNetwork(const uint8_t* ptr_nnData,
                size_t nnDataLen,
                size_t tensorArenaSize = 1000000);

  /**
   * Get the number of input layers.
   *
   * @returns The number of input layers
   */
  size_t getNumInputLayers() const;

  /**
   * Get the size of an input layer.
   *
   * @param[in] inputLayerIdx Index of the input layer
   *
   * @returns The size of the input layer
   */
  size_t getInputLayerSize(size_t inputLayerIdx) const;

  /**
   * Get the number of output layers
   *
   * @returns The number of output layers
   */
  size_t getNumOutputLayers() const;

  /**
   * Get the size of an output layer.
   *
   * @param[in] outputLayerIdx Index of the output layer
   *
   * @returns The size of the output layer
   */
  size_t getOutputLayerSize(size_t outputLayerIdx) const;

  /**
   * Set the input data of the neural network.
   *
   * @param[in] inputLayerIdx Index of the input layer
   * @param[in] ptr_inputData Input data to set
   * @param[in] inputDataLen Size of the input data
   */
  void setInputData(size_t inputLayerIdx,
                    const float* ptr_inputData,
                    size_t inputDataLen);

  /**
   * Execute the neural network.
   */
  void execute();

  /**
   * Get the output data of the neural network.
   */
  void getOutputData(size_t outputLayerIdx,
                     float* ptr_outputData,
                     size_t outputDataLen) const;
};
