"""Classes for remote procedure calls (RPCs).

File:   communication/remote_procedure_call.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2022-05-24


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

import threading
import time
import datetime
from abc import ABCMeta, abstractmethod
from typing import Union


class RemoteProcedureCall(metaclass=ABCMeta):
    """Abstract base class for remote procedure calls (RPCs)."""

    # Status constants
    RUNNING = 0
    SUCCEEDED = 1
    FAILED = 2

    @abstractmethod
    def get_status(self) -> int:
        """Get the current status of the RPC.

        Returns:
            - _: int
                  Status code.
        """

        raise NotImplementedError

    @abstractmethod
    def get_result(self) -> Union[Message, list[Message]]:
        """Get the result of the RPC.

        This method blocks until the results are available.

        Returns:
            - _: Union[Message. list[Message]]
                  Response messages from LExCI minions.
        """

        raise NotImplementedError


class DirectRemoteProcedureCall(RemoteProcedureCall):
    """A direct remote procedure call (RPC) to a single recipient."""

    def __init__(
        self, mailbox: Mailbox, cmd_msg: Message, job_timeout: float
    ) -> None:
        """Initialize the remote procedure call and send/execute it right away.

        Arguments:
            - mailbox: Mailbox
                  Mailbox to a LExCI minion.
            - cmd_msg: Message
                  Command message.
            - job_timeout: float (Unit: s)
                  Timeout for the minion job.
        """

        super().__init__()

        self._mailbox = mailbox
        self._cmd_msg = cmd_msg
        self._job_timeout = job_timeout

        self._resp_msg = None
        self._status = RemoteProcedureCall.RUNNING

        self._lock = threading.Lock()
        self._t_start = datetime.datetime.now()
        self._thread = threading.Thread(target=self._function, daemon=True)
        self._thread.start()

    def _function(self) -> None:
        """Send the command message and wait for an answer."""

        try:
            self._mailbox.send(self._cmd_msg)

            while True:
                resp_msg = self._mailbox.receive()
                if resp_msg is not None:
                    if resp_msg.payload["ref"] != self._cmd_msg.id:
                        raise RuntimeError("Invalid response.")
                    break

                t_elapsed = (
                    datetime.datetime.now() - self._t_start
                ).total_seconds()
                if t_elapsed >= self._job_timeout:
                    raise RuntimeError("Minion job timed out.")
                time.sleep(0.01)

            self._lock.acquire()
            self._resp_msg = resp_msg
            self._status = RemoteProcedureCall.SUCCEEDED
            self._lock.release()
        except:
            self._lock.acquire()
            self._status = RemoteProcedureCall.FAILED
            self._lock.release()

    def get_status(self) -> int:
        """Get the current status of the RPC.

        Returns:
            - _: int
                  Status code.
        """

        self._lock.acquire()
        status = self._status
        self._lock.release()

        return status

    def get_result(self) -> Message:
        """Get the response message with the results.

        This method blocks until the worker thread has terminated. If the RPC
        status is not `RemoteProcedureCall.SUCCEEDED`, the return value is
        `[None]`.

        Returns:
            - _: list[Message]
                  Response message from the LExCI minion.
        """

        if self._thread is not None:
            self._thread.join()
            self._thread = None

        return self._resp_msg


class BroadcastRemoteProcedureCall(RemoteProcedureCall):
    """Remote procedure call that is broadcast to multiple LExCI minions."""

    def __init__(
        self, mailboxes: list[Mailbox], cmd_msg: Message, job_timeout: float
    ) -> None:
        """Initialize the remote procedure call and broadcast/execute it right
        away.

        Arguments:
            - mailboxes: list[Mailbox]
                  Mailboxes to LExCI minions.
            - cmd_msg: Message
                  Command message.
            - job_timeout: float (Unit: s)
                  Timeout for the minion job.
        """

        super().__init__()

        self._mailboxes = mailboxes
        self._cmd_msg = cmd_msg
        self._job_timeout = job_timeout

        self._rpcs = []
        for mailbox in self._mailboxes:
            self._rpcs.append(
                DirectRemoteProcedureCall(
                    mailbox, self._cmd_msg, self._job_timeout
                )
            )

    def get_status(self) -> int:
        """Get the current status of the broadcast RPC.

        Returns:
            - _: int
                  Status code.
        """

        # Get the current status of all RPCs
        statuses = self.get_detailed_status()

        if any([e == RemoteProcedureCall.RUNNING for e in statuses]):
            return BroadcastRemoteProcedureCall.RUNNING
        elif any([e == RemoteProcedureCall.FAILED for e in statuses]):
            return BroadcastRemoteProcedureCall.FAILED
        else:
            return BroadcastRemoteProcedureCall.SUCCEEDED

    def get_detailed_status(self) -> list[int]:
        """Get the status of each individual remote procedure call.

        Returns:
            - _: list[int]
                  Status codes.
        """

        statuses = []
        for rpc in self._rpcs:
            statuses.append(rpc.get_status())
        return statuses

    def get_result(self) -> list[Message]:
        """Get the response messages with the results.

        This method blocks until all individual remote procedure calls have
        terminated.

        Returns:
            - _: list[Message]
                  Response messages from the LExCI minions.
        """

        resp_msgs = []
        for rpc in self._rpcs:
            resp_msg = rpc.get_result()  # Waits for the RPC to finish
            if rpc.get_status() == RemoteProcedureCall.SUCCEEDED:
                resp_msgs.append(resp_msg)

        return resp_msgs
