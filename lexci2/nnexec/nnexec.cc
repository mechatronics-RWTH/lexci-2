/**
 * C99 wrapper for executing a TensorFlow Lite Micro neural network.
 *
 * @file:   nnexec.cc
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

#include "nnexec.h"
#include "NeuralNetwork.hh"

void*
createNeuralNetwork(const uint8_t* ptr_nnData,
                    size_t nnDataLen,
                    size_t tensorArenaSize)
{
  return static_cast<void*>(
    new NeuralNetwork(ptr_nnData, nnDataLen, tensorArenaSize));
}

void
destroyNeuralNetwork(void* ptr_nn)
{
  if (ptr_nn != nullptr) {
    delete static_cast<NeuralNetwork*>(ptr_nn);
  }
}

size_t
getNumInputLayers(void* ptr_nn)
{
  return static_cast<NeuralNetwork*>(ptr_nn)->getNumInputLayers();
}

size_t
getInputLayerSize(void* ptr_nn, size_t inputLayerIdx)
{
  return static_cast<NeuralNetwork*>(ptr_nn)->getInputLayerSize(inputLayerIdx);
}

size_t
getNumOutputLayers(void* ptr_nn)
{
  return static_cast<NeuralNetwork*>(ptr_nn)->getNumOutputLayers();
}

size_t
getOutputLayerSize(void* ptr_nn, size_t outputLayerIdx)
{
  return static_cast<NeuralNetwork*>(ptr_nn)->getOutputLayerSize(
    outputLayerIdx);
}

void
setInputData(void* ptr_nn,
             size_t inputLayerIdx,
             const float* ptr_inputData,
             size_t inputDataLen)
{
  static_cast<NeuralNetwork*>(ptr_nn)->setInputData(
    inputLayerIdx, ptr_inputData, inputDataLen);
}

void
execute(void* ptr_nn)
{
  static_cast<NeuralNetwork*>(ptr_nn)->execute();
}

void
getOutputData(void* ptr_nn,
              size_t outputLayerIdx,
              float* ptr_outputData,
              size_t outputDataLen)
{
  static_cast<NeuralNetwork*>(ptr_nn)->getOutputData(
    outputLayerIdx, ptr_outputData, outputDataLen);
}
