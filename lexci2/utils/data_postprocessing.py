"""Helper functions for post-processing LExCI data.

File:   lexci2/utils/data_postprocessing.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2022-12-07


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


import lexci2
from lexci2.data_containers import Experience, Episode, Cycle


def downsample_episode(episode: Episode, dt: float) -> Episode:
    """Downsample an episode where the agent was executed less frequently than
    the rest of the model.

    It is assumed that the agent's inputs and outputs are held until the next
    execution.

    Arguments:
        - episode: Episode
              The episode to process. The original data is not modified.
        - dt: float (Unit: s)
              Time between agent executions.

    Returns:
        - _: Episode
              A deep-copied, downsampled version of the episode.

    Raises:
        - ValueError
    """

    # Ensure that all experiences have timestamps
    for experience in episode:
        if experience.t is None:
            raise ValueError("Found an experience without a timestamp.")
    # Check whether the episode is empty
    if len(episode) == 0:
        raise ValueError("The episode contains no experiences.")

    # Create container for downsampled experiences. The very first experiences
    # is always kept.
    downsampled_episode = Episode(episode.agent_id)
    downsampled_episode.append_experience(episode[0])

    # Sample down
    ref_i = 0
    for i in range(1, len(episode)):
        if episode[i].t - episode[ref_i].t >= dt:
            downsampled_episode.append_experience(episode[i])
            ref_i = i

    return downsampled_episode


def downsample_cycle(cycle: Cycle, dt: float) -> Cycle:
    """Downsample a cycle where the agent was executed less frequently that the
    rest of the model.

    It is assumed that the agent's inputs and outputs are held until the next
    execution.

    Arguments:
        - cycle: Cycle
              The cycle to process. The original data is not modified.
        - dt: float (Unit: s)
              Time between agent executions.

    Returns:
        - _: Cycle
              A deep-copied, downsampled version of the cycle.

    Raises:
        - ValueError
    """

    downsampled_cycle = Cycle()
    for episode in cycle:
        downsampled_episode = downsample_episode(episode, dt)
        downsampled_cycle.add_episode(downsampled_episode)
    return downsampled_cycle
