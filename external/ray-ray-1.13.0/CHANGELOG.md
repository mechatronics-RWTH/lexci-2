# Change Log


## Modified Files
- **rllib/agents/ddpg/noop_model.py**

  Added a try-except-block to `NoopModel.forward()` to prevent crashes in case
  `input_dict["obs_flat"]` doesn't have the key 'obs'.


## New Files
- **rllib/agents/ddpg/lexci_ddpg.py**

  Wrote a version of the `DDPGTrainer` which allows users to choose between
  training on replay data and training on a given batch.
