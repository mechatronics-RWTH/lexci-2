# Writing a Minion

As explained in the
[paper](https://link.springer.com/article/10.1007/s10489-024-05573-0), every new
RL problem LExCI shall be applied to necessitates its own Minion. Let's see why
that is the case and how to go about writing a Minion.


## The Data Generation Domain

While the Universal LExCI Masters are configured using only primitive data
types, i.e. quantities that can easily be passed through a file, the Minion must
be given the logic for interacting with the embedded system and other pieces of
hardware in the environment. That's because the makeup of its data generation
domain is unique to each problem.

Starting with the embedded device itself, the manner of access varies from
manufacturer to manufacturer and sometimes from model to model. For instance,
the
[MicroAutoBox III](https://www.dspace.com/en/pub/home/products/hw/micautob/microautobox3.cfm),
a rapid control prototyping (RCP) system, is manipulated via the proprietary
control application
[ControlDesk](https://www.dspace.com/en/pub/home/products/sw/experimentandvisualization/controldesk.cfm). On the other end of the spectrum, a
[Raspberry Pi 5](https://www.raspberrypi.com/products/raspberry-pi-5/)[^1] needs
no such program as it can — being quite powerful and featuring a conventional
Linux distribution — directly host the Minion. In the majority of cases where
communication between the device and the LExCI Minion is channelled through some
control software, the Minion must know how to use the program's
[API](https://en.wikipedia.org/wiki/API) for automation.

As for the hardware components in the data generation domain, they may have to
be initialised, triggered, or reset at certain points of the training process
(e.g. before starting an episode). The necessary steps highly depend on the part
in question and the RL problem. The same is true for software models.


[^1]: Admittedly, the Raspberry Pi is more akin to a personal computer (PC) than
      most traditional embedded systems; some might even argue that it shouldn't
      be regarded as one. It satisfies all criteria, though, when acting as a
      controller in a larger system, so let's just agree that it *can* be an
      embedded device.


## Callback Functions

TODO


## Software Controllers

TODO
