"""Abstract base class for all agents.

File:   lexci2/agents/agent.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2022-04-20


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

from lexci2.lexci_env import LexciEnvConfig, LexciEnv
from lexci2.data_containers import Cycle
from lexci2.lexci_input_reader import LexciInputReader
from lexci2.neural_network_modules.neural_network_module import (
    NeuralNetworkModule,
)

import ray
from ray.tune.logger import UnifiedLogger
from ray.rllib.policy.sample_batch import SampleBatch
import tensorflow as tf
from keras.engine.functional import Functional

import copy
import uuid
import os
import re
from abc import ABCMeta, abstractmethod
import numpy as np
from typing import Any, Callable, Optional


def lexci_logger_creator(
    log_dir: str,
) -> Callable[[dict[str, Any]], UnifiedLogger]:
    """Create a logger function which generates its output files in a specified
    folder.

    Arguments:
        - log_dir: str
              Directory where the log files shall be saved.

    Returns:
        - _: Callable[[dict[str, Any]], UnifiedLogger]
              Creator function for a `UnifiedLogger` which logs to a
              predetermined directory.
    """

    def logger_creator(config: dict[str, Any]) -> UnifiedLogger:
        """Create a `UnifiedLogger` whose outputs are written to a specific
        folder.

        Arguments:
            - config: dict[str, Any]
                  Logger configuration.

        Returns:
            - _: UnifiedLogger
                  Logger that writes to a specified directory.
        """

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return UnifiedLogger(config, log_dir, loggers=None)

    return logger_creator


def nn_modifying_method(
    func: Callable[["Agent", ...], Any]
) -> Callable[["Agent", ...], Any]:
    """Wrapper for methods that change an agent's neural network.

    This ensures that `self._update_nn_module()` is always called after
    `func()`.

    Arguments:
        - func: Callable[[Agent, ...], Any]
              The method to wrap.

    Returns:
        - _: Callable[[Agent, ...], Any]
              The wrapped method.
    """

    def wrapped_func(self: "Agent", *args, **kwargs) -> None:
        """Wrapped function.

        Arguments:
            - self: Agent
                  An `Agent` object.
            - args:
                  Positional arguments.
            - kwargs:
                  Keyword arguments.

        Returns:
            - _: Any
                  The return value of `func`.
        """

        result = func(self, *args, **kwargs)
        self._update_nn_module()
        return result

    return wrapped_func


class Agent(metaclass=ABCMeta):
    """A trainable agent."""

    def __init__(
        self,
        id: str,
        env_config: LexciEnvConfig,
        trainer_config: dict[str, Any],
        log_dir: str = "~/lexci_results/ray_results",
    ) -> None:
        """Initialize the agent.

        Arguments:
            - id: str
                  ID of the agent.
            - env_config: LexciEnvConfig
                  Configuration of the agent's environment.
            - trainer_config: dict[str, Any]
                  Configuration of the agent's trainer.
            - log_dir: str (default: '~/lexci_results/ray_results')
                  Folder to write the trainer's logs to.
        """

        self._id = id
        self._env_config = copy.deepcopy(env_config)
        self._trainer_config = copy.deepcopy(trainer_config)
        self._log_dir = os.path.expanduser(log_dir)

        self._lexci_env_name = "LexciEnv_" + self._id
        self._trainer = None
        self._cycle = [None]  # List acts as a reference to the current cycle

        # Modify the trainer config for LExCI's mode of operation
        self._trainer_config["num_workers"] = 0
        self._trainer_config["rollout_fragment_length"] = self._trainer_config[
            "train_batch_size"
        ]
        self._trainer_config["framework"] = "tf2"
        self._trainer_config["input"] = lambda _: LexciInputReader(self)

        # Register the environment
        ray.tune.register_env(
            self._lexci_env_name, lambda _: LexciEnv(env_config)
        )

        # Placeholder for the neural network module which sub-classes of `Agent`
        # must set at the end of their initializers by invoking
        # `self._update_nn_module()`
        self._nn_module = None

    def set_cycle_no(self, cycle_no: int) -> None:
        """Explicitly set the iteration number of the agent's trainer so that it
        corresponds with the current cycle number.

        Arguments:
            - cycle_no: int
                  Current cycle number.
        """

        self._trainer._iteration = cycle_no

    @abstractmethod
    def get_models(self) -> dict[str, Functional]:
        """Get all models of the agent, i.e. not only its policy NN but also
        value function approximators etc.

        Returns:
            - _: dict[str, Functional]:
                  A dictionary with all models of the agent.
        """

        raise NotImplementedError

    @abstractmethod
    def set_models(self, new_models: dict[str, Functional]) -> None:
        """Set all models of the agent, i.e. not only its policy NN but also
        value function approximators etc.

        Arguments:
            - new_models: dict[str, Functional]
                  A dictionary containing the new models of the agent.

        Raises:
            - ValueError:
                  - If `models` is incomplete.
        """

        raise NotImplementedError

    def import_models(self, model_folder_name: str) -> None:
        """Import all models of an agent from h5-files in a specific folder.

        Arguments:
            - model_folder_name: str
                  Path to the folder containing the h5-files.
        """

        model_folder_name = os.path.abspath(model_folder_name)
        models = self.get_models()

        # Import the model files
        imported_models = {}
        for k, v in models.items():
            if v is None:
                model = None
            else:
                model_file_name = os.path.join(model_folder_name, f"{k}.h5")
                model = tf.keras.models.load_model(model_file_name)
            imported_models[k] = model

        # Overwrite the agent's models
        self.set_models(imported_models)

    def export_models(self, model_folder_name: str) -> None:
        """Export all models of the agent as h5-files.

        Arguments:
            - model_folder_name: str
                  Path to the folder where the h5-files shall be stored.

        Raises:
            - ValueError:
                  - If the folder `model_folder_name` already exists.
        """

        # Create the folder
        model_folder_name = os.path.abspath(model_folder_name)
        if os.path.exists(model_folder_name):
            raise ValueError(
                f"The folder '{model_folder_name}' already exists."
            )
        os.makedirs(model_folder_name)

        # Export the individual models of the agent
        models = self.get_models()
        for k, v in models.items():
            # Make sure that the model is not `None`. This is important because
            # some algorithms have optional models (e.g. DDPG's twin Q-model).
            if v is not None:
                model_file_name = os.path.join(model_folder_name, f"{k}.h5")
                v.save(model_file_name, save_format="h5")

    @abstractmethod
    def get_nn(self) -> Functional:
        """Get the current policy neural network of the agent.

        This method may have to be overwritten for algorithms that don't store
        the policy NN in `base_model`.

        Returns:
            - _: Functional
                  Neural network of the agent.
        """

        raise NotImplementedError

    def get_nn_module(self) -> NeuralNetworkModule:
        """Get the neural network module of the agent.

        Returns:
            - _: NeuralNetworkModule
                  The neural network module of the agent.
        """

        return self._nn_module

    @abstractmethod
    def _update_nn_module(self) -> None:
        """Update the neural network module.

        This method must be invoked after every training step.
        """

        raise NotImplementedError

    def get_id(self) -> str:
        """Get the agent's ID.

        Returns:
            - _: str
                  ID of the agent
        """

        return self._id

    def get_env_config(self) -> LexciEnvConfig:
        """Get the agent's LExCI environment configuration.

        Returns:
            - _: LexciEnvConfig
                  Environment configuration of the agent.
        """

        return copy.deepcopy(self._env_config)

    def get_trainer_config(self) -> dict[str, Any]:
        """Get the agent's trainer configuration.

        Returns:
            - _: dict[str, Any]
                  Deep-copied trainer configuration of the agent.
        """

        return copy.deepcopy(self._trainer_config)

    def get_num_training_experiences(self) -> int:
        """Get the number of experiences the agent needs for a training step.

        Returns:
            - _: int
                  The number of experiences that is required for training.
        """

        return self._trainer_config["train_batch_size"]

    def set_log_dir(self, log_dir: str) -> None:
        """Set the trainer's logging directory.

        Arguments:
            - log_dir: str
                  Folder to write the trainer's logs to.
        """

        self._log_dir = os.path.expanduser(log_dir)
        self._trainer._create_logger(
            self._trainer.config, lexci_logger_creator(self._log_dir)
        )

    @staticmethod
    @abstractmethod
    def get_default_trainer_config() -> dict[str, Any]:
        """Get the default configuration of the trainer.

        Returns:
            - _: dict[str, Any]
                  Default configuration.
        """

        raise NotImplementedError

    def save_checkpoint_file(self, checkpoint_dir: str) -> None:
        """Save the trainer's current state as a checkpoint file.

        Arguments:
            - checkpoint_dir: str
                  Path to the folder where the checkpoint shall be saved.
        """

        self._trainer.save_checkpoint(checkpoint_dir)

    def load_checkpoint_file(self, checkpoint_file: str) -> None:
        """Restore another trainer state by loading a checkpoint file.

        Arguments:
            - checkpoint_file: str
                  Path to the checkpoint file to restore.
        """

        # Load the checkpoint
        self._trainer.load_checkpoint(checkpoint_file)

        # Set the trainer's iteration number
        m = re.match(
            "^checkpoint-(?P<cycle_no>\d+)$", os.path.basename(checkpoint_file)
        )
        self._trainer._iteration = int(m["cycle_no"])

    def export_nn_to_bytes(self, fmt: str) -> bytes:
        """Export the agent's neural network as bytes.

        Arguments:
            - fmt: str
                  Format of the returned data. Must be either 'keras' or
                  'tflite'.

        Returns:
            - _: bytes
                  Bytes of the neural network.

        Raises:
            - ValueError
        """

        if fmt == "keras":
            # Create a temporary file export
            while True:
                temp_file_name = f"{uuid.uuid4()}.h5"
                if not os.path.exists(temp_file_name):
                    break
            self.export_nn_to_file(temp_file_name, "keras")

            # Read the bytes of the temp file
            data = b""
            with open(temp_file_name, "rb") as f:
                data = f.read()

            # Remove the temporary file and return the bytes
            if os.path.exists(temp_file_name):
                os.remove(temp_file_name)
            return data
        elif fmt == "tflite":
            tflite_converter = tf.lite.TFLiteConverter.from_keras_model(
                self.get_nn()
            )
            tflite_converter.experimental_new_converter = False
            tflite_converter.experimental_quantizer = False
            tflite_converter.experimental_enable_resource_variables = False
            return tflite_converter.convert()
        else:
            raise ValueError(f"Unknown format '{fmt}'.")

    def export_nn_to_file(self, file_name: str, fmt: str) -> None:
        """Export the agent's policy neural network to a file.

        Arguments:
          - file_name: str
              Name of the output file.
          - fmt: str
              Format of the exported data. Must be either 'keras' or 'tflite'.

        Raises:
          - ValueError
        """

        if fmt == "keras":
            self.get_nn().save(file_name, save_format="h5")
            with open(file_name, "rb+") as f:
                s = f.read().replace(b'"_initializer"', b'"RandomNormal"')
                f.seek(0)
                f.truncate(0)
                f.write(s)
        elif fmt == "tflite":
            tflite_converter = tf.lite.TFLiteConverter.from_keras_model(
                self.get_nn()
            )
            tflite_converter.experimental_new_converter = False
            tflite_converter.experimental_quantizer = False
            tflite_converter.experimental_enable_resource_variables = False
            model = tflite_converter.convert()
            with open(file_name, "wb") as f:
                f.write(model)
        else:
            raise ValueError(f"Unknown format '{fmt}'.")

    @abstractmethod
    def _create_batch(self, cycle: Cycle) -> SampleBatch:
        """Postprocess cycle data and convert it into a `SampleBatch`.

        Arguments:
            - cycle: Cycle
                  Training cycle data.

        Returns:
            - _: SampleBatch
                  Preprocessed training cycle data.
        """

        raise NotImplementedError

    @nn_modifying_method
    def train(self, cycle: Cycle) -> None:
        """Train the agent with training cycle data.

        Arguments:
            - cycle: Cycle
                  Training cycle data.

        Raises:
            - ValueError
        """

        # The current training data is copied here. The input reader will set
        # the value to `None` after the cycle has been processed.
        self._cycle[0] = copy.deepcopy(cycle)

        # TODO: Perform checks

        # Filter training episodes
        # TODO: Currently, no filtering is performed.
        """
        eps = []
        for e in cycle.eps:
            if e.agent_id == self._id:
                eps.append(e)
        cycle = Cycle(eps)
        """

        # Train
        self._trainer.train()
