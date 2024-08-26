"""Agent that utilizes Twin Delayed DDPG (TD3) for training.

File:   lexci2/agents/td3_agent.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2024-08-08


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

from lexci2.agents.ddpg_agent import DdpgAgent

import ray.rllib.agents.ddpg.td3 as td3

import copy
from typing import Any


class Td3Agent(DdpgAgent):
    """Agent that trains with TD3."""

    @staticmethod
    def get_default_trainer_config() -> dict[str, Any]:
        """Get the default configuration of the trainer.

        Returns:
            - _: dict[str, Any]:
                  Default configuration.
        """

        return copy.deepcopy(td3.TD3_DEFAULT_CONFIG)
