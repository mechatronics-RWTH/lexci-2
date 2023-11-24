"""Minion of the (L)earning and (Ex)periencing (C)ycle (I)nterface (LExCI).

File:   minion/minion.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2022-05-25


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


from lexci2.communication.mailbox import Mailbox
from lexci2.communication.message import Message
from lexci2.data_containers import Cycle

import time
import logging
from typing import Callable, Any


logger = logging.getLogger(__name__)


class Minion:
    """Minion that reports to a LExCI master and collects training data."""

    def __init__(
        self,
        master_addr: str,
        master_port: int,
        training_func: Callable[[bytes, int, int, dict[str, Any]], Cycle],
        validation_func: Callable[[bytes, int, dict[str, Any]], Cycle],
        mailbox_buffer_size: int = 1 * 1024**3,
    ) -> None:
        """Initialize the Minion.

        Arguments:
            - master_addr:
                  IP-address of the LExCI Master.
            - master_port:
                  Listening port of the LExCI Master.
            - training_func: Callable[[bytes, int, dict[str, Any]], Cycle]
                  Callback for generating training data.

                  Arguments:
                      - _: bytes:
                            Bytes of the current neural network.
                      - _: int:
                            Current cycle number.
                      - _: int
                            Number of experiences to generate.
                      - _: dict[str, Any]
                            Minion parameters provided by the Master.

                  Returns:
                      - _: Cycle
                            Training cycle data.
            - validation_func: Callable[[bytes, int, dict[str, Any]], Cycle]
                  Callback for generating validation data.

                  Arguments:
                      - _: bytes
                            Bytes of the current neural network.
                      - _: int
                            Current cycle number.
                      - _: dict[str, Any]
                            Minion parameters provided by the Master.

                  Returns:
                      - _: Cycle
                            Validation cycle data.
            - mailbox_buffer_size: int (Default: 1 * 1024**2, Unit: B)
                  Buffer size of the minion's `Mailbox`.
        """

        super().__init__()

        self._master_addr = master_addr
        self._master_port = master_port
        self._mailbox_buffer_size = mailbox_buffer_size
        self._training_func = training_func
        self._validation_func = validation_func

        self._mailbox = None

    def mainloop(self) -> None:
        """Main loop of the Minion."""

        while True:
            try:
                logger.info(
                    "Trying to connect to"
                    + f" {self._master_addr}:{self._master_port}..."
                )
                self._mailbox = Mailbox.from_address(
                    self._master_addr,
                    self._master_port,
                    self._mailbox_buffer_size,
                )
                self._mailbox.start()
                logger.info("Connection established.")

                while True:
                    # Receive a command message
                    cmd_msg = self._mailbox.wait_and_receive()
                    logger.info("Received a command message.")
                    cmd = cmd_msg.payload["cmd"]
                    nn_bytes = bytes(cmd_msg.payload["nn_bytes"])
                    cycle_no = cmd_msg.payload["cycle_no"]
                    minion_params = cmd_msg.payload["minion_params"]

                    # Perform the task
                    if cmd == "perform_training_run":
                        logger.info("Performing a training run.")
                        num_experiences = cmd_msg.payload["num_experiences"]
                        cycle = self._training_func(
                            nn_bytes, cycle_no, num_experiences, minion_params
                        )
                    elif cmd == "perform_validation_run":
                        logger.info("Performing a validation run.")
                        cycle = self._validation_func(
                            nn_bytes, cycle_no, minion_params
                        )
                    else:
                        logger.warn("Unknown command.")
                        raise RuntimeError("Unknown command.")

                    # Send results back
                    logger.info("Sending results back.")
                    resp_msg = Message(
                        {"ref": cmd_msg.id, "cycle_json": cycle.to_json()}
                    )
                    self._mailbox.send(resp_msg)
            except:
                if self._mailbox is not None:
                    self._mailbox.stop()
                    self._mailbox = None
                time.sleep(10.0)
