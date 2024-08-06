"""Listener of the LExCI master server.

File: master/listener.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2022-05-20


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

from lexci2.utils.continuous_thread import ContinuousThread
from lexci2.communication.mailbox import Mailbox

import socket
import logging


logger = logging.getLogger(__name__)


class Listener(ContinuousThread):
    """Continuous thread responsible for accepting incoming LExCI minion
    connections.
    """

    def __init__(self, addr: str, port: int, mailbox_buffer_size: int) -> None:
        """Initialize the listener.

        Arguments:
            - addr: str
                  IP-address of the server.
            - port: int
                  Listening port.
            - mailbox_buffer_size: int (Unit: B)
                  Mailbox buffer size.
        """

        super().__init__()

        self._addr = addr
        self._port = port
        self._mailbox_buffer_size = mailbox_buffer_size

        self._sock = None
        self._num_requested_minions = 0
        self._minions = []

    def _open_listening_socket(self) -> None:
        """Open the listening socket."""

        if self._sock is None:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind((self._addr, self._port))
            self._sock.listen()
            logger.info(f"Listening on {self._addr}:{self._port}.")

    def _close_listening_socket(self) -> None:
        """Close the listening socket."""

        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except:
                pass
            self._sock.close()
            self._sock = None
            logger.info("Stopped listening.")

    def stop(self) -> None:
        """Close the listening socket and stop the continuous thread.."""

        self._close_listening_socket()
        super().stop()

    def function(self) -> None:
        """Listen for incoming connection requests by LExCI minions."""

        self.lock()
        requested_minions = self._num_requested_minions - len(self._minions)
        self.unlock()

        if requested_minions > 0:
            self._open_listening_socket()
            (
                conn,
                client_addr,
            ) = self._sock.accept()  # TODO: Is this non-blocking?

            mailbox = Mailbox(conn, self._mailbox_buffer_size)
            mailbox.start()
            self.lock()
            self._minions.append(mailbox)
            logger.info(
                f"Accepted connection from {client_addr[0]}:{client_addr[1]}."
            )
            self.unlock()
        else:
            self._close_listening_socket()

    def request_minions(self, num_minions: int) -> None:
        """Request a certain number of minions.

        If the listener has already accepted more minions than are requested,
        the excess connections are terminated.

        Arguments:
            - num_minions: int
                  Number of new requested minions.
        """

        self.lock()

        self._num_requested_minions = max(0, num_minions)

        # Remove excess minion connections
        if len(self._minions) > self._num_requested_minions:
            excess_minions = self._minions[self._num_requested_minions :]
            self._minions = self._minions[: self._num_requested_minions]

            for mailbox in excess_minions:
                mailbox.stop()

        self.unlock()

    def get_minions(self) -> list[Mailbox]:
        """Get new accepted minions.

        Returns:
            - _: list[Mailbox]
                  Accepted minions.
        """

        self.lock()
        minions = self._minions.copy()
        self._minions.clear()
        self._num_requested_minions = max(
            0, self._num_requested_minions - len(minions)
        )
        self.unlock()

        return minions
