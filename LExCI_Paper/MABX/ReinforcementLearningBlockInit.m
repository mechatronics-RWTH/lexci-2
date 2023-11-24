%% Summary
% File:    ReinforcementLearningBlockInit.m
% Date:    2022-07-28
% Authors: Lucas Koch     (koch_luc@mmp.rwth-aachen.de)
%          Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
%
%
% Copyright 2023 Teaching and Research Area Mechatronics in Mobile Propulsion,
%                RWTH Aachen University
%
% Licensed under the Apache License, Version 2.0 (the "License"); you may not
% use this file except in compliance with the License. You may obtain a copy of
% the License at: http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
% WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
% License for the specific language governing permissions and limitations under
% the License.

%% RL Settings
rl_algorithm = 1; % 1: PPO 2: DDPG
action_type = 1; % 1: continuous 2: discrete (not implemented yet)
action_sampling_mode = 1; % 1: tanh 2: clip

num_actions = 1; % number of actions

%% S-Function Builder Parameters
inc{1} = '/home/kevin/WIP/MABX_Pendulum_Env/external_cpp_libs';
inc{2} = '/usr/local/MATLAB/R2022b/extern/include';
inc{3} = '';
setappdata(0, 'SfunctionBuilderIncludePath', inc);

%% Initial neural network data
INITIAL_NN_BYTES = uint8(zeros(1, 65536));

%% internal settings - DO NOT CHANGE

if rl_algorithm == 1 % PPO
    action_mean_selector_ppo = 1:num_actions;
    action_std_selector_ppo = (num_actions+1):(2*num_actions);
    action_selector_ddpg = 1:num_actions;
elseif rl_algorithm == 2 % DDPG
    action_mean_selector_ppo = 1:num_actions;
    action_std_selector_ppo = 1:num_actions;
    action_selector_ddpg = 1:num_actions;   
end
