"""Classes for communicating via TCP/IP.

File:   communication/tcp_ip_communication.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2023-04-26


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

import socket
import threading
import logging


logger = logging.getLogger(__name__)


class _Receiver(ContinuousThread):
    """Continuous thread for receiving data."""

    def __init__(
        self,
        connection_lock: threading.Lock,
        connection_sock: socket.socket,
        buffer_size: int = 65536,
        period: float = 0.01,
    ) -> None:
        """Initialize the receiver.

        Arguments:
            - connection_lock: threading.Lock
                  Reference to the `TcpIpConnection`'s lock.
            - connection_sock: socket.socket
                  Reference to the `TcpIpConnection`'s socket.
            - buffer_size: int (Default: 65536, Unit: B)
                  Read buffer size.
            - period: float (Default: 0.01, Unit: s)
                  Periodic time of the thread's main loop.
        """

        super().__init__(period)

        self._connection_lock = connection_lock
        self._connection_sock = connection_sock

        self._buffer_size = buffer_size
        self._buffer = b""

    def function(self) -> None:
        """Non-blocking function that takes care of its own exception handling
        and is continuously called by the thread.
        """

        self.lock()
        buffer_len = len(self._buffer)
        self.unlock()

        self._connection_lock.acquire()
        try:
            data = self._connection_sock.recv(self._buffer_size - buffer_len)
        except:
            pass
        self._connection_lock.release()
        self.lock()
        self._buffer += data
        self.unlock()

    def receive(self) -> bytes:
        """Read data.

        Returns:
            - _: bytes
                  The data that has been received.
        """

        self.lock()
        data = self._buffer
        self._buffer = b""
        self.unlock()

        return data


class _Sender(ContinuousThread):
    """Continuous thread for sending data."""

    def __init__(
        self,
        connection_lock: threading.Lock,
        connection_sock: socket.socket,
        period: float = 0.01,
    ) -> None:
        """Initialize the sender.

        Arguments:
            - connection_lock: threading.Lock
                  Reference to the `TcpIpConnection`'s lock.
            - connection_sock: socket.socket
                  Reference to the `TcpIpConnection`'s socket.
            - period: float (Default: 0.01, Unit: s)
                  Periodic time of the thread's main loop.
        """

        super().__init__(period)

        self._connection_lock = connection_lock
        self._connection_sock = connection_sock

        self._buffer = b""

    def function(self) -> None:
        """Non-blocking function that takes care of its own exception handling
        and is continuously called by the thread.
        """

        self.lock()
        data = self._buffer
        self._buffer = b""
        self.unlock()

        self._connection_lock.acquire()
        try:
            self._connection_sock.sendall(data)
        except:
            logger.warn("Failed to send data.")
        self._connection_lock.release()

    def send(self, data: bytes) -> None:
        """Write data.

        Arguments:
            - data: bytes
                  The data to send.
        """

        self.lock()
        self._buffer += data
        self.unlock()


class TcpIpConnection:
    """Helper class for communicating via a TCP/IP socket."""

    def __init__(
        self,
        sock: socket.socket,
        buffer_size: int = 65536,
        period: float = 0.01,
    ) -> None:
        """Initialize the connection.

        Arguments:
            - sock: socket.socket
                  The socket to use for communication. The `TcpIpConnection`
                  object will take exclusive ownership of the socket, i.e. it's
                  henceforth responsible for e.g. closing it.
            - buffer_size: int (Default: 65536, Unit: B)
                  Size of the receive buffer.
            - period: float (Default: 0.01, Unit: s)
                  Periodic time of the receiver's and sender's main loops.
        """

        self._lock = threading.Lock()

        self._sock = sock
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_size)
        self._sock.setblocking(False)

        self._receiver = _Receiver(self._lock, self._sock, buffer_size, period)
        self._sender = _Sender(self._lock, self._sock, period)

    @classmethod
    def connect_to(
        cls,
        host_addr: str,
        host_port: int,
        buffer_size: int = 65536,
        period: float = 0.01,
    ) -> None:
        """Factory method that first creates the socket and connects to the
        specified address.

        Arguments:
            - host_addr: str
                  Address to connect to, i.e. the address of the server.
            - host_port: int
                  Port number to connect to, i.e. the listening port of the server.
            - buffer_size: int (Default: 65536, Unit: B)
                  Size of the receive buffer.
            - period: float (Default: 0.01, Unit: s)
                  Periodic time of the receiver's and sender's main loops.
        """

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host_addr, host_port))
        except:
            logger.error(f"Could not connect to {host_addr}:{host_port}.")
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except:
                pass
            sock.close()
            raise

        logger.info(f"Connection to {host_addr}:{host_port} established.")
        return cls(sock, buffer_size, period)

    @classmethod
    def listen_on(
        cls,
        addr: str,
        port: int,
        buffer_size: int = 65536,
        period: float = 0.01,
    ) -> None:
        """Factory method that first listens for an incoming connection.

        Arguments:
            - addr: str
                  Address to listen on.
            - host_port: int
                  Port number to listen on.
            - buffer_size: int (Default: 65536, Unit: B)
                  Size of the receive buffer.
            - period: float (Default: 0.01, Unit: s)
                  Periodic time of the receiver's and sender's main loops.
        """

        # Create a socket for listening
        listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listening_sock.bind((addr, port))
        listening_sock.listen()
        logger.info(f"Listening for an incoming connection on {addr}:{port}.")

        # Wait for a connection request to arrive
        sock, client_addr = listening_sock.accept()
        logger.info(
            f"Accepted a connection from {client_addr[0]}:{client_addr[1]}."
        )

        # Close the listening socket
        try:
            listening_sock.shutdown(socket.SHUT_RDWR)
        except:
            pass
        listening_sock.close()
        logger.info(f"Stopped listening on {addr}:{port}.")

        # Create the `TcpIpConnection` object
        return cls(sock, buffer_size, period)

    def __del__(self) -> None:
        """Delete the object."""

        self.terminate()

    def terminate(self) -> None:
        """Terminate the connection."""

        if self._receiver is not None:
            self._receiver.stop()
            self._receiver = None
        if self._sender is not None:
            self._sender.stop()
            self._sender = None
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except:
                pass
            self._sock.close()
            self._sock = None

    def receive(self) -> bytes:
        """Receive data over the TCP/IP connection.

        Returns:
            - _: bytes
                  The received data.
        """

        return self._receiver.receive()

    def send(self, data: bytes) -> None:
        """Send data over the TCP/IP connection.

        Arguments:
            - _: bytes
                  The data to send.
        """

        self._sender.send(data)


class TcpIpServer(ContinuousThread):
    """Server that accepts incoming TCP/IP connections."""

    def __init__(
        self,
        addr: str,
        port: int,
        max_num_connections: Optional[int] = None,
        buffer_size: int = 65536,
        period: float = 0.01,
    ) -> None:
        """Initialize the server.

        Arguments:
            - addr: str
                  IP-address of the server.
            - port: int
                  Port to listen on.
            - max_num_connections: Optional[int] (Default: None)
                  Maximum number of connections to accept.
            - buffer_size: int (Default: 65536, Unit: B)
                  Buffer size of the accepted connections.
            - period: float (Default: 0.01, Unit: s)
                  Periodict time of the threads main loop. This value will also
                  be set for the `TcpIpConnection` objects.
        """

        super().__init__(period)

        # Copy arguments
        self._addr = addr
        self._port = port
        self._max_num_connections = max_num_connections
        self._buffer_size = buffer_size
        self._period = period

        # Buffer for accepted connections
        self._num_accepted_connections = 0
        self._connections = []

        # Socket for listening
        self._listening_sock = None

    def _open_listening_socket(self) -> None:
        """Open the socket and listen for incoming connections."""

        if self._listening_sock is None:
            self._listening_sock = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM
            )
            self._listening_sock.setsockopt(
                socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
            )
            self._listening_sock.setblocking(False)
            self._listening_sock.bind((addr, port))
            self._listening_sock.listen()
            logger.info(f"Listening for incoming connections on {addr}:{port}.")

    def _close_listening_socket(self) -> None:
        """Close the listening socket."""

        if self._listening_sock is not None:
            try:
                self._listening_sock.shutdown(socket.SHUT_RDWR)
            except:
                pass

            self._listening_sock.close()
            self._listening_sock = None
            logger.info("Stopped listening.")

    def function(self) -> None:
        """Non-blocking function that takes care of its own exception handling
        and is continuously called by the thread.
        """

        if self._num_accepted_connections < self._max_num_connections:
            # Open the listening socket if that hasn't happened yet
            self._open_listening_socket()

            try:
                # Wait for a connection request to arrive
                sock, client_addr = listening_sock.accept()
                logger.info(
                    "Accepted a connection from"
                    + f" {client_addr[0]}:{client_addr[1]}."
                )
                self._num_accepted_connections += 1
                connection = TcpIpConnection(
                    sock, self._buffer_size, self._period
                )
                self.lock()
                self._connections.append(connection)
                self.unlock()
            except OSError as e:
                if e.errno in [errno.EAGAIN, errno.EWOULDBLOCK]:
                    pass
                else:
                    raise
        else:
            self._close_listening_socket()

    def get_accepted_connections(self) -> list[TcpIpConnection]:
        """Get the list of accepted connections.

        Returns:
            - _: list[TcpIpConnection]
                  List of accepted connections.
        """

        self.lock()
        connections = self._connections
        self._connections = []
        self.unlock()
        return connections
