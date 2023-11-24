"""Class for handling LExCI messages.

File:   communication/mailbox.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2022-05-06


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
from lexci2.communication.message import Message

import uuid
import time
import datetime
import copy
import socket
import threading
import logging
from typing import Union


logger = logging.getLogger(__name__)


class Mailbox:
    """Class for sending and receiving LExCI messages through threaded
    procedures.
    """

    # Chunk size for data transmission
    _CHUNK_SIZE = 65536

    def __init__(
        self,
        sock: socket.socket,
        buffer_size: int,
        period: float = 0.01,
        heartbeat_interval: float = 10.0,
        heartbeat_timeout: float = 300.0,
    ) -> None:
        """Initialize the mailbox.

        `Mailbox` takes ownership of the socket it uses and is responsible for
        closing it. Once stopped, it therefore cannot be restarted.

        Arguments:
            - sock: socket.socket
                  Socket for communication.
            - buffer_size: int (Unit: B)
                  Size of the receiver's data buffer.
            - period: float (Default: 0.01, Unit: s)
                  Periodic time of the sender and receiver.
            - heartbeat_interval: float (Default: 10.0, Unit: s)
                  Time in between heartbeats.
            - heartbeat_timeout: float (Default: 300.0, Unit: s)
                  Time after which the connection is considered dead if no
                  heartbeat has been received.
        """

        self._sock = sock
        self._buffer_size = buffer_size
        self._period = period
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_timeout = heartbeat_timeout

        self._sock.setblocking(False)

        self._id = str(uuid.uuid4())
        self._lock = threading.Lock()
        self._recv_queue = []
        self._send_queue = []

        self._receiver = _Receiver(
            self._period,
            self._sock,
            self._buffer_size,
            self._heartbeat_timeout,
            self._recv_queue,
            self._lock,
        )
        self._sender = _Sender(
            self._period,
            self._sock,
            self._heartbeat_interval,
            self._send_queue,
            self._lock,
        )

    @classmethod
    def from_address(
        cls,
        host_addr: str,
        host_port: int,
        buffer_size: int,
        period: float = 0.01,
        heartbeat_interval: float = 10.0,
        heartbeat_timeout: float = 300.0,
    ) -> "Mailbox":
        """Factory method that provides a socket by connecting to a host first.

        Arguments:
            - host_addr: str
                  Host address.
            - host_port: int
                  Port of the host.
            - buffer_size: int (Unit: B)
                  Size of the data buffer.
            - period: float (Default: 0.01, Unit: s)
                  Periodic time of the sender and receiver.
            - heartbeat_interval: float (Default: 10.0, Unit: s)
                  Time in between heartbeats.
            - heartbeat_timeout: float (Default: 300.0, Unit: s)
                  Time after which the connection is considered dead if no
                  heartbeat has been received.
        """

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host_addr, host_port))
            return cls(
                sock, buffer_size, period, heartbeat_interval, heartbeat_timeout
            )
        except:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except:
                pass
            sock.close()
            sock = None
            raise

    def get_id(self) -> str:
        """Get the mailbox ID.

        Returns:
            - _: str
                  ID of the mailbox.
        """

        return self._id

    def get_local_addr(self) -> tuple[str, int]:
        """Get the local address of the Mailbox.

        Returns:
            - addr: str
                  Local IP-address.
            - port: int
                  Local port number.
        """

        return self._sock.getsockname()

    def get_peer_addr(self) -> tuple[str, int]:
        """Get the address of the Mailbox's peer.

        Returns:
            - addr: str
                  IP-address of the peer.
            - port: int
                  Port number of the peer.
        """

        return self._sock.getpeername()

    def start(self) -> "None":
        """Start the threads for sending and receiving."""

        self._receiver.start()
        self._sender.start()

    def stop(self) -> None:
        """Stop the worker threads and close the socket."""

        # Stop the worker threads
        self._sender.stop()
        self._receiver.stop()

        # Close the socket
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except:
            pass
        self._sock.close()
        self._sock = None

    def is_connection_alive(self) -> bool:
        """Check if the connection is alive.

        Returns:
            - _: bool
                  `True` if the connection is up, else `False`.
        """

        return self._receiver.is_running() and self._sender.is_running()

    def receive(self) -> Union[Message, None]:
        """Get and dequeue the first received message.

        Returns:
            - _: Union[Message, None]
                  Received message or `None` if the queue is empty.

        Raises:
            - RuntimeError
        """

        if not self.is_connection_alive():
            raise RuntimeError("Connection down.")

        msg = None
        self._lock.acquire()
        if len(self._recv_queue) > 0:
            msg = self._recv_queue.pop(0)
        self._lock.release()
        return msg

    def wait_and_receive(self) -> Message:
        """Wait for a message to arrive and receive it.

        Returns:
            - _: Message
                  Received message.
        """

        while True:
            msg = self.receive()
            if msg is not None:
                return msg
            time.sleep(self._period)

    def send(self, msg: Message) -> None:
        """Enqueue a message for sending.

        Arguments:
            - msg: Message
                  Message to send.

        Raises:
            - RuntimeError
        """

        if not self.is_connection_alive():
            raise RuntimeError("Connection down.")

        self._lock.acquire()
        self._send_queue.append(copy.deepcopy(msg))
        self._lock.release()


class _Receiver(ContinuousThread):
    """Continuous thread for receiving messages inside a `Mailbox`."""

    def __init__(
        self,
        period: float,
        sock: socket.socket,
        buffer_size: int,
        heartbeat_timeout: float,
        recv_queue: list[Message],
        mailbox_lock: threading.Lock,
    ) -> None:
        """Initialize the receiver.

        Arguments:
            - period: float (Unit: s)
                  Periodic time of the main loop.
            - sock: socket.socket
                  Socket for commuication.
            - buffer_size: int (Unit: B)
                  Size of the data buffer.
            - heartbeat_timeout: float (Unit: s)
                  Time after which the connection is considered dead if no
                  heartbeat has been received.
            - recv_queue: list[Message]
                  Queue to write received messages to.
            - mailbox_lock: threading.Lock
                  Lock for accessing the socket and the queue.
        """

        super().__init__(period)

        self._sock = sock  # Reference to the mailbox's socket
        self._buffer_size = buffer_size
        self._heartbeat_timeout = heartbeat_timeout
        self._recv_queue = recv_queue  # Reference to the mailbox's queue
        self._mailbox_lock = mailbox_lock  # Reference to the mailbox's lock

        self._recv_buf = b""
        self._t_heartbeat = (
            datetime.datetime.now()
        )  # TODO: Is this an okay initial value?

    def function(self) -> None:
        """Receive messages and store them in a queue."""

        # Try to retrieve data from the socket
        data = b""
        self._mailbox_lock.acquire()
        try:
            t_recv_start = datetime.datetime.now()
            while True:
                data = self._sock.recv(self._buffer_size - len(self._recv_buf))
                if data != b"":
                    self._recv_buf += data
                    self._t_heartbeat = datetime.datetime.now()

                t_now = datetime.datetime.now()
                b_interrupt_recv = (
                    t_now - t_recv_start
                ).total_seconds > self._heartbeat_timeout / 2
                if len(data) == 0 or b_interrupt_recv:
                    break
        except:
            pass
        self._mailbox_lock.release()

        # Attempt to re-assemble the message
        if len(self._recv_buf) >= 5:
            msg_len = int.from_bytes(
                self._recv_buf[1:5], byteorder="big", signed=False
            )
            if len(self._recv_buf) >= msg_len:
                try:
                    msg = Message.from_bytes(self._recv_buf[:msg_len])
                    self._recv_buf = self._recv_buf[msg_len:]
                    if msg.payload != {}:  # Do not save heartbeats
                        self._mailbox_lock.acquire()
                        self._recv_queue.append(msg)
                        self._mailbox_lock.release()
                except:
                    logger.warning(
                        "Failed to decode a message. Receiver buffer reset."
                    )
                    self._recv_buf = b""

        # Check if the last heartbeat has timed out
        t = datetime.datetime.now()
        if (t - self._t_heartbeat).total_seconds() >= self._heartbeat_timeout:
            logger.error("Heartbeat timed out. Connection lost.")
            raise Exception


class _Sender(ContinuousThread):
    """Continuous thread for sending messages inside a `Mailbox`."""

    def __init__(
        self,
        period: float,
        sock: socket.socket,
        heartbeat_interval: float,
        send_queue: list[Message],
        mailbox_lock: threading.Lock,
    ) -> None:
        """Initialize the sender.

        Arguments:
            - period: float (Unit: s)
                  Periodic time of the main loop.
            - sock: socket.socket
                  Socket for communication.
            - heartbeat_interval: float (Unit: s)
                  Time in between heartbeats.
            - send_queue: list[Message]
                  Queue containing the messages to send.
            - mailbox_lock: threading.Lock
                  Lock for accessing the socket and the queue.
        """

        super().__init__(period)

        self._sock = sock  # Reference to the mailbox's socket
        self._heartbeat_interval = heartbeat_interval
        self._send_queue = send_queue  # Reference to the mailbox's queue
        self._mailbox_lock = mailbox_lock  # Reference to the mailbox's lock

        self._t_heartbeat = datetime.datetime.now()

    def _sendall(self, data: bytes) -> None:
        """Custom implementation of `socket.sendall()` that works with
        non-blocking sockets.

        This method does not check the state of the mailbox's lock.

        Arguments:
            - data: bytes
                  Data to send.
        """

        n_sent = 0
        n_total = len(data)
        while n_sent != n_total:
            try:
                n_chunk = n_sent + Mailbox._CHUNK_SIZE
                if n_chunk > n_total:
                    n_chunk = n_total

                n_sent += self._sock.send(data[n_sent:n_total])
            except BlockingIOError:
                pass

    def function(self) -> None:
        """Send queued messages."""

        # Fetch the messages that shall be sent
        self._mailbox_lock.acquire()
        queue = copy.deepcopy(self._send_queue)
        self._send_queue.clear()
        self._mailbox_lock.release()

        # Transmit all entries in the queue
        for msg in queue:
            try:
                self._mailbox_lock.acquire()
                self._sendall(msg.to_bytes())
                self._t_heartbeat = datetime.datetime.now()
            except:
                logger.error("Failed to send a message. Connection lost.")
                raise
            finally:
                self._mailbox_lock.release()

        # See if it's time to send a heartbeat
        t = datetime.datetime.now()
        if (t - self._t_heartbeat).total_seconds() >= self._heartbeat_interval:
            try:
                self._mailbox_lock.acquire()
                self._sendall(Message({}).to_bytes())
                self._t_heartbeat = t
            except:
                logger.error("Failed to send a heartbeat. Connection lost.")
                raise
            finally:
                self._mailbox_lock.release()
