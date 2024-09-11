# First Training

Whether you want to familiarise yourself with LExCI or you'd like to verify that
everything went smoothly during
[installation](https://github.com/mechatronics-RWTH/lexci-2/blob/main/documentation/01_installation.md),
the pendulum swing-up problem is a great place to start. It's the environment
that was tackled in the
[LExCI paper](https://link.springer.com/article/10.1007/s10489-024-05573-0) and,
as such, it naturally comes with pre-configured hyperparameter sets as well as
an implementation of the Minion. All you need to do is to start the toolchain
and wait for the agent to converge. This document will show you how.


## Configuration Files

LExCI is configured through [YAML](https://yaml.org/) files that contain the
general settings of the framework as well as the RL agent's hyperparameters.
Among other things, the former is made up of the dimensions of the observation
space, the action space size, networking settings (e.g. IP address and port
number of the Master), how often validations are performed, etc. The latter
comprises the architecture of the agent's neural network(s), the learning rate
(schedule), and the like. When going through them, you'll see that a block
comment separates the (in our experience) most relevant parameters from those of
minor importance. Please consult
[Ray/RLlib's manual](https://docs.ray.io/en/releases-1.13.0/rllib/rllib-algorithms.html)
for the ones in the second group: some parameters may have no effect depending
on the chosen RL algorithm, others may be overridden by related options, and
still others might be internally overwritten by LExCI.

Templates for each supported RL algorithm can be found in the folders of LExCI's
[Universal Masters](https://github.com/mechatronics-RWTH/lexci-2/tree/main/lexci2/universal_masters).
Here, we'll use configuration files that have already been fine-tuned for the
[pendulum environment](https://github.com/mechatronics-RWTH/lexci-2/tree/main/lexci2/test_envs/pendulum_minion).


## Starting the Master and the Minion

Open a terminal, activate the virtual environment, and run
`Lexci2UniversalDdpgMaster /path/to/pendulum_env_ddpg_config.yaml` to start the
Master. After a couple of seconds (and many, many log messages), you should see
a line stating that it is listening for incoming connections.

Open a second terminal, activate the virtual environment, and `cd` to
`lexci2/test_envs/pendulum_minion` within your local copy of the repository.
There, execute `python3.9 pendulum_minion.py ddpg` in order to start the Minion.
Both Master and Minion should report that a connection has been established and
the Minion should start printing a wall of text.


## Tracking the Agent's Progress

The output of the Minion shows the state of the pendulum environment that is
being simulated:
1. The first column represents the (denormalised) observation of the agent, i.e.
   the x- and y-coordinate of the pendulum's free end as well as its angular
   velocity.
2. The second column is the chosen (denormalised) action, that is the torque
   which is applied to the base of the the pendulum.
3. The reward of the experience is printed in the third column.
4. The final column shows the action distribution. In the case of DDPG, this is
   simply the mean of the Gaussian distribution from which the actions are
   sampled. The standard deviation is internally calculated as a function of the
   training cycle.

Considering the definition of the pendulum swing-up problem, a good agent is
expected to quickly move the rod to the upright position (the x- and
y-coordinate of the pendulum are close to 1 and 0, respectively) and hold it
still (the angular velocity is (almost) 0) without exerting too much torque (the
action should be as close to 0 as possible).

LExCI stores all of the training-related data on the hard drive. By default, the
output is saved in `~/lexci_results`. There, you'll find a file named `log.csv`
which documents the training progression. After roughly 50 cycles, you should
see that the maximum episode reward has become greater than -10 and that all
three reward metrics have gone up and plateaued, albeit with some occasional
outliers. Depending on your hardware, the agent takes approximately 45-60
minutes to converge.
