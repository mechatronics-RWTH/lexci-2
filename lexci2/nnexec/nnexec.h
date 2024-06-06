/**
 * C99 wrapper for executing a TensorFlow Lite Micro neural network.
 *
 * @file:   nnexec.h
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

#ifndef NNEXEC_H
#define NNEXEC_H

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

#include <stddef.h>
#include <stdint.h>

  /**
   * Create a neural network object.
   *
   * Since C is an imperative language, the created object must be returned
   * using a `void*` pointer.
   *
   * @param[in] ptr_nnData Byte data of the neural network
   * @param[in] nnDataLen Size of the neural network data [1]
   * @param[in] tensorArenaSize Size of the tensor arena [B]
   * @returns Pointer to the neural network object
   */
  void* createNeuralNetwork(const uint8_t* ptr_nnData,
                            size_t nnDataLen,
                            size_t tensorArenaSize);

  /**
   * Destroy a neural network object.
   *
   * @param[in] ptr_nn Pointer to the neural network object to destroy
   */
  void destroyNeuralNetwork(void* ptr_nn);

  /**
   * Get the number of input layers of a neural network.
   *
   * @returns The number of input layers
   */
  size_t getNumInputLayers(void* ptr_nn);

  /**
   * The the size of a neural network's input layer.
   *
   * @returns The size of the input layer
   */
  size_t getInputLayerSize(void* ptr_nn, size_t inputLayerIdx);

  /**
   * Get the number of output layers of a neural network.
   *
   * @returns The number of output layers
   */
  size_t getNumOutputLayers(void* ptr_nn);

  /**
   * Get the size of a neural network's output layer.
   *
   * @returns The size of the output layer
   */
  size_t getOutputLayerSize(void* ptr_nn, size_t outputLayerIdx);

  /**
   * Set the input of the neural network.
   *
   * @param[in] ptr_nn Pointer to the neural network whose input shall be set
   * @param[in] inputLayerIdx Index (starting at 0) of the input to set. Though
   *            virtually all neural networks have just a single input layer,
   * this argument allows users to work with ones that have more.
   * @param[in] ptr_inputData Input data to set
   * @param[in] inputDataLen Input data length
   */
  void setInputData(void* ptr_nn,
                    size_t inputLayerIdx,
                    const float* ptr_inputData,
                    size_t inputDataLen);

  /**
   * Execute a neural network.
   *
   * The results must be retrieved using the `getOutputData()` function.
   *
   * @param[in] ptr_nn Pointer to the neural network to execute
   */
  void execute(void* ptr_nn);

  /**
   * Get the output of the neural network.
   *
   * @param[in] ptr_nn Pointer to the neural network to get the output from
   * @param[in] outputLayerIdx Index (starting at 0) of the output to get.
   * Though most neural networks have just a single output layer, this argument
   * allows users to work with ones that have more.
   * @param[out] ptr_outputData Memory to write the output of the neural network
   *             to. It must already have been allocated and it must have the
   *             correct size.
   * @param[in] Output data length
   */
  void getOutputData(void* ptr_nn,
                     size_t outputLayerIdx,
                     float* ptr_outputData,
                     size_t outputDataLen);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* NNEXEC_H */
