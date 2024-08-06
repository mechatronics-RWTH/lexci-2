"""Agent that utilizes an off-policy algorithm for training.

File:   lexci2/agents/off_policy_agent.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2022-11-08


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

from lexci2.agents.agent import Agent, nn_modifying_method
from lexci2.data_containers import Cycle

from ray.rllib.policy.sample_batch import SampleBatch

from abc import abstractmethod


class OffPolicyAgent(Agent):
    """An agent that trains using an off-policy algorithm."""

    def train(self, cycle: Cycle) -> None:
        """Train the agent with training cycle data.

        Arguments:
            - cycle: Cycle
                  Training cycle data.
        """

        self.train_on_cycle(cycle, True)

    @abstractmethod
    @nn_modifying_method
    def train_on_cycle(
        self, cycle: Cycle, b_add_to_memory: bool = True
    ) -> None:
        """Train on experiences from a given cycle.

        This method allows the agent to train on on-policy data only, i.e. on
        experiences that were collected using the latest version of the policy
        network.

        Arguments:
            - cycle: Cycle
                  A LExCI cycle.
            - b_add_to_memory: bool (default: True)
                  Whether the experiences shall be added to the agent's replay
                  memory after training is done.
        """

        raise NotImplementedError

    @abstractmethod
    @nn_modifying_method
    def train_on_replay_memory(self) -> None:
        """Train using only experiences from the replay memory.

        Implementations must invoke `self._update_nn_module()` after training.
        """

        raise NotImplementedError

    @abstractmethod
    def add_lexci_cycle_to_replay_memory(self, cycle: Cycle) -> None:
        """Add experiences to the agent's replay memory.

        Arguments:
            - cycle: Cycle
                  A LExCI cycle containing the experiences that shall be added
                  to the replay buffer of the agent.
        """

        raise NotImplementedError

    @abstractmethod
    def add_sample_batch_to_replay_memory(
        self, sample_batch: SampleBatch
    ) -> None:
        """Add experiences to the agent's replay memory.

        Arguments:
            - sample_batch: SampleBatch
                  A `SampleBatch` containing the data that shall be added to the
                  replay buffer of the agent.
        """

        raise NotImplementedError

    @abstractmethod
    def clear_replay_memory(self) -> None:
        """Clear the replay buffer of the agent."""

        raise NotImplementedError

    @abstractmethod
    def get_replay_memory_size(self) -> int:
        """Get the number of experiences in the replay memory buffer.

        Returns:
          - _: int
              Number of experiences in the replay memory.
        """

        raise NotImplementedError
