# Introduction

TODO


## Motivation

Embedded systems/devices are specialised computers which are integrated into
larger systems where they act as control units. They are typically characterised
by being physically small, offering adequate performance for the intended task
without drawing too much power, a rugged design that allows them to operate in
harsh conditions, and a real-time operating system which guarantees that jobs
are completed within a fixed time frame. Despite their size, embedded devices
play a major role in keeping our modern world moving: they are the electronic
control units (ECUs) in our vehicles, the chips that regulate traffic signals,
or the computers inside aircraft.

With the emergence of artificial intelligence (AI) and its ever-increasing
prevalence in so many areas of life, the question naturally arose whether AI
methods could be utilised to learn control functions, i.e. the software that
runs on embedded systems. Reinforcement learning (RL) is particularly
interesting in that regard as it generates its own training data; a RL agent
freely interacts with its environment to collect experiences which are then used
to optimise its policy (that is, the control function). As a result, little
human input is needed to obtain the finished software compared to traditional
development approaches.

Considering all this, the combination of embedded devices and RL might sound
like a match made in Heaven. But there's a catch: established machine learning
(ML) and RL libraries are often incompatible with the inherent limitations of
embedded systems (e.g. the lack of disk space, memory, and computing power).
Consequently, scientists and engineers have been compelled to implement
algorithms and data structures (for example, neural networks (NNs)) themselves
in the past to ensure that they would work on the targetted device. In a sense,
people had to reinvent the wheel and they've done that successfully as research
papers prove. Now, it would be unfair to expect a handful of engineers to
write full-fledged ML/RL libraries *on top* of their actual jobs; one has to
acknowledge, though, that these isolated solutions generally fall short of what
professional software features.


## The LExCI Framework

The *Learning and Experiencing Cycle Interface* or LExCI for short was developed
as a mediator between the realm of embedded computing and established ML/RL
libraries. Specifically, the framework uses
[Ray/RLlib](https://github.com/ray-project/ray) to train agents and
[TensorFlow](https://github.com/tensorflow/tensorflow)/[TensorFlow Lite Micro](https://github.com/tensorflow/tflite-micro)
to model their neural networks. TODO.


## Modus Operandi

TODO
