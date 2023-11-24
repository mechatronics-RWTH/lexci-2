"""A special version of the `DDPGTrainer` that can choose to only sample from
the workers or the replay memory buffer. Thus, it fits neatly into LExCI's
workflow.

File:   ray/rllib/agents/ddpg/lexci_ddpg.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2022-10-13


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


from ray.rllib.agents.ddpg import DDPGTrainer
from ray.rllib.agents.trainer import Trainer
from ray.rllib.utils.metrics import SYNCH_WORKER_WEIGHTS_TIMER
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.execution.train_ops import ( train_one_step,
    multi_gpu_train_one_step )
from ray.rllib.utils.annotations import ExperimentalAPI
from ray.rllib.utils.metrics import ( NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED, TARGET_NET_UPDATE_TIMER )
from ray.rllib.utils.typing import ResultDict
from ray.rllib.utils.metrics import LAST_TARGET_UPDATE_TS, NUM_TARGET_UPDATES
from ray.rllib.utils.annotations import ExperimentalAPI, override, PublicAPI
from ray.rllib.policy.sample_batch import SampleBatch

import logging




logger = logging.getLogger(__name__)




class LexciDdpgTrainer(DDPGTrainer):
  """A special version of the `DDPGTrainer` that can choose to exclusively
  sample from workers or its replay memory."""

  # Training modes
  # In normal mode, `LexciDdpgTrainer` operates just like `DDPGTrainer`, i.e. it
  # samples both from its workers and its replay memory.
  NORMAL_MODE = 0
  # Sample from the replay memory only.
  REPLAY_MEMORY_ONLY_MODE = 1
  # Only sample from a given batch.
  GIVEN_BATCH_ONLY_MODE = 2



  @PublicAPI
  def __init__(
      self,
      config: "Optional[Union[PartialTrainerConfigDict, TrainerConfig]]" = None,
      env: "Optional[Union[str, EnvType]]" = None,
      logger_creator: "Optional[Callable[[], Logger]]" = None,
      remote_checkpoint_dir: "Optional[str]" = None,
      sync_function_tpl: "Optional[str]" = None,
  ):
    super().__init__(config, env, logger_creator, remote_checkpoint_dir,
        sync_function_tpl)
    self._mode = LexciDdpgTrainer.NORMAL_MODE
    self._given_batch = None



  @ExperimentalAPI
  @override(Trainer)
  def training_iteration(self) -> ResultDict:
    """An adaptation of the original method that can decide whether to sample
    from workers or from the replay memory buffer.

    Returns:
      - _: ResultDict:
          Result dictionary of the training iteration.
    """
    
    batch_size = self.config["train_batch_size"]
    local_worker = self.workers.local_worker()

    if self._mode == LexciDdpgTrainer.NORMAL_MODE:
      # Sample n MultiAgentBatches from n workers.
      new_sample_batches = synchronous_parallel_sample(
          worker_set=self.workers, concat=False
      )
      for batch in new_sample_batches:
          # Update sampling step counters.
          self._counters[NUM_ENV_STEPS_SAMPLED] += batch.env_steps()
          self._counters[NUM_AGENT_STEPS_SAMPLED] += batch.agent_steps()
          # Store new samples in the replay buffer
          self.local_replay_buffer.add_batch(batch)
      # Sample data from the replay memory
      train_batch = self.local_replay_buffer.replay()
    elif self._mode == LexciDdpgTrainer.REPLAY_MEMORY_ONLY_MODE:
      # Sample data from the replay memory
      train_batch = self.local_replay_buffer.replay()
    elif self._mode == LexciDdpgTrainer.GIVEN_BATCH_ONLY_MODE:
      train_batch = self._given_batch
    else:
      raise ValueError("Unknown mode.")

    # Train on the collected experiences
    if self.config.get("simple_optimizer") is True:
        train_results = train_one_step(self, train_batch)
    else:
        train_results = multi_gpu_train_one_step(self, train_batch)

    # Update the target network
    cur_ts = self._counters[NUM_ENV_STEPS_SAMPLED]
    last_update = self._counters[LAST_TARGET_UPDATE_TS]
    if cur_ts - last_update >= self.config["target_network_update_freq"]:
      with self._timers[TARGET_NET_UPDATE_TIMER]:
        to_update = local_worker.get_policies_to_train()
        local_worker.foreach_policy_to_train(
            lambda p, pid: pid in to_update and p.update_target()
        )
      self._counters[NUM_TARGET_UPDATES] += 1
      self._counters[LAST_TARGET_UPDATE_TS] = cur_ts

    # Update weights and global variables
    global_vars = {
        "timestep": self._counters[NUM_ENV_STEPS_SAMPLED],
    }
    with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
      self.workers.sync_weights(global_vars=global_vars)

    # Return the metrics of the training iteration
    return train_results



  def train_on_given_batch(self, batch: SampleBatch,
      b_add_to_memory: bool = True) -> None:
    """Train on experiences from a specific batch.

    Arguments:
      - batch: SampleBatch
          A sample batch.
      - b_add_to_memory: bool (default: `True`)
          Whether the experiences shall be added to the replay memory after
          training is done.
    """

    self._mode = LexciDdpgTrainer.GIVEN_BATCH_ONLY_MODE
    self._given_batch = batch

    self.train()
    if b_add_to_memory:
      self.add_to_replay_memory(batch)

    self._given_batch = None
    self._mode = LexciDdpgTrainer.NORMAL_MODE



  def train_on_replay_memory(self) -> None:
    """Train by using experiences from the replay memory buffer only."""

    self._mode = LexciDdpgTrainer.REPLAY_MEMORY_ONLY_MODE
    self.train()
    self._mode = LexciDdpgTrainer.NORMAL_MODE



  def add_to_replay_memory(self, batch: SampleBatch) -> None:
    """Add experiences to the replay memory.

    Arguments:
      - batch: SampleBatch
          Sample batch to add to the replay memory.
    """

    self.local_replay_buffer.add_batch(batch)



  def get_replay_memory_size(self) -> int:
    """Get the number of experiences in the replay memory buffer.

    Returns:
      - _: int
          Number of experiences in the replay memory.
    """

    return self.local_replay_buffer.get_state()["num_added"]

