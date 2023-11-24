"""Reader for converting LExCI's data into a Ray-compatible format.

The design is a bit convoluted as it has to bridge the gap between how Ray calls
the reader and the way LExCI handles information. `LexciInputReader` provides
the interface that is expected by the RL-library while internally invoking the
calling agent's own batch generation method on its own reference to the training
data. That way, one can replace the training data during operation without
creating a new trainer and supplement experiences with specific quantities that
an RL-algorithm may ask for.

File:   lexci2/lexci_input_reader.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2022-04-21


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


from lexci2.data_containers import Cycle

from ray.rllib.offline import InputReader
from ray.rllib.utils.typing import SampleBatchType

import copy


class LexciInputReader(InputReader):
    """Custom reader for the data LExCI provides.

    It removes the agent's cycle data (i.e. sets it to `None`) after the first
    call to `next()`.
    """

    def __init__(self, agent: "Agent") -> None:
        """Initialize the input reader.

        Arguments:
            - agent: Agent
                  Parent agent.
        """

        self._agent = agent

    def next(self) -> SampleBatchType:
        """Get the data in a Ray-compatible format.

        Returns:
            - _: SampleBatchType
                  Sample batch with all the additional information the agent
                  requires for training.

        Raises:
            - ValueError
        """

        if self._agent._cycle[0] is None:
            raise ValueError(
                "No (more) data available! Make sure that LExCI cycles provide"
                + " enough experiences to perform a training step on them."
            )

        b = self._agent._create_batch(self._agent._cycle[0])
        self._agent._cycle[0] = None
        return b
