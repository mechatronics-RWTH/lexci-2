# First Training

Whether you want to familiarise yourself with LExCI or you'd like to verify that
everything went smoothly during
[installation](https://github.com/mechatronics-RWTH/lexci-2/blob/main/documentation/01_installation.md),
the pendulum swing-up problem is a great place to start. It's the environment
that was tackled in the
[LExCI paper](https://link.springer.com/article/10.1007/s10489-024-05573-0) and,
as such, it naturally comes with pre-configured hyperparameter sets as well as
an implementation of the Minion. All you need to do is to generate a
configuration file, start the Master and the Minion, and wait until the agent
converges. This document will show you how.


## Configuration File

LExCI is configured through [JSON](https://www.json.org/json-en.html) files that
contain the general settings of the framework as well as the RL agent's
hyperparameters. To aid users with their generation, the repository provides so
called config creators, i.e. Python scripts which list the Master's options and
expose the most important hyperparameters of the chosen RL algorithm.

Open a terminal, activate LExCI's virtual environment, and `cd` to your local
copy of the repository. Next, type `cd lexci2/tests/pendulum_minion` and then
run `python3.9 ddpg_config_creator.py ddpg_config.json`. This will create a file
named `ddpg_config.json` in the current working directory.


## Starting the Master and the Minion

Open a terminal, activate the virtual environment, and run
`Lexci2UniversalDdpgMaster /path/to/ddpg_config.json` to start the Master. After
a couple of seconds (and many, many log messages), you should see a line stating
that it is listening for incoming connections.

Open a second terminal, activate the virtual environment, and `cd` to
`lexci2/tests/pendulum_minion` within your local copy of the repository. There,
execute `python3.9 pendulum_minion.py ddpg` in order to start the Minion. Both
Master and Minion should report that a connection has been established and the
Minion should start printing a wall of text.


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
