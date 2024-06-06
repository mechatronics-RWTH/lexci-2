/**
 * Helper class for creating and executing TensorFlow Lite Micro neural
 * networks.
 *
 * @file:   NeuralNetwork.c
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

#include "NeuralNetwork.hh"

#include <cstring>

NeuralNetwork::NeuralNetwork(const uint8_t* ptr_nnData,
                             size_t nnDataLen,
                             size_t tensorArenaSize)
{
  // Copy the bytes of the neural network
  sptr_nnData = std::unique_ptr<uint8_t>(new uint8_t[nnDataLen]);
  std::memcpy(sptr_nnData.get(), ptr_nnData, nnDataLen);

  // Create the error reporter
  sptr_microErrorReporter = std::unique_ptr<tflite::MicroErrorReporter>(
    new tflite::MicroErrorReporter());

  // Create the resolver
  sptr_resolver =
    std::unique_ptr<tflite::AllOpsResolver>(new tflite::AllOpsResolver());

  // Allocate memory for the tensor arena
  sptr_tensorArena = std::unique_ptr<uint8_t>(new uint8_t[tensorArenaSize]);

  // Create the interpreter
  sptr_interpreter = std::unique_ptr<tflite::MicroInterpreter>(
    new tflite::MicroInterpreter(::tflite::GetModel(sptr_nnData.get()),
                                 *sptr_resolver,
                                 sptr_tensorArena.get(),
                                 tensorArenaSize,
                                 sptr_microErrorReporter.get()));
  sptr_interpreter->AllocateTensors();
}

size_t
NeuralNetwork::getNumInputLayers() const
{
  return sptr_interpreter->inputs_size();
}

size_t
NeuralNetwork::getInputLayerSize(size_t inputLayerIdx) const
{
  // An input layer with n nodes has shape 1xn. Hence, `data[0] = 1` and
  // `data[1] = n`.
  return sptr_interpreter->input(inputLayerIdx)->dims->data[1];
}

size_t
NeuralNetwork::getNumOutputLayers() const
{
  return sptr_interpreter->outputs_size();
}

size_t
NeuralNetwork::getOutputLayerSize(size_t outputLayerIdx) const
{
  // An output layer with n nodes has shape 1xn. Hence, `data[0] = 1` and
  // `data[1] = n`.
  return sptr_interpreter->output(outputLayerIdx)->dims->data[1];
}

void
NeuralNetwork::setInputData(size_t inputLayerIdx,
                            const float* ptr_inputData,
                            size_t inputDataLen)
{
  TfLiteTensor* ptr_inputLayer = sptr_interpreter->input(inputLayerIdx);

  for (size_t i = 0; i < inputDataLen; i++) {
    ptr_inputLayer->data.f[i] = ptr_inputData[i];
  }
}

void
NeuralNetwork::execute()
{
  sptr_interpreter->Invoke();
}

void
NeuralNetwork::getOutputData(size_t outputLayerIdx,
                             float* ptr_outputData,
                             size_t outputDataLen) const
{
  TfLiteTensor* ptr_outputLayer = sptr_interpreter->output(outputLayerIdx);

  for (size_t i = 0; i < outputDataLen; i++) {
    ptr_outputData[i] = ptr_outputLayer->data.f[i];
  }
}
