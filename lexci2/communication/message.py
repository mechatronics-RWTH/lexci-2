"""Container for messages that are exchanged between LExCI Master and Minion.

File:   communication/message.py
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


import copy
import uuid
import json
from typing import Any


class Message:
    """Container for the data that is exchanged between LExCI's Master and its
    Minions.
    """

    def __init__(self, payload: dict[str, Any]) -> None:
        """Initialize the message.

        Arguments:
            - payload: dict[str, Any]
                  JSON-serializable dictionary containing the payload of the
                  message.
        """

        self.id = int(uuid.uuid4())
        self.payload = copy.deepcopy(payload)

    def __str__(self) -> str:
        """Convert the `Message` into a string.

        Returns:
            - _: str
                  Human-readable string representation of the message.
        """

        return f"0x{self.id:X} ({len(self.to_bytes())} B): {self.payload}"

    @classmethod
    def from_bytes(cls, encoded_msg: bytes) -> "Message":
        """Decode a `Message` in byte format.

        Arguments:
            - encoded_msg: bytes
                  Encoded message bytes.

        Raises:
            - ValueError
        """

        # Perform checks
        if len(encoded_msg) < 22:
            raise ValueError(
                "The encoded message isn't large enough to contain a complete"
                + " LExCI message."
            )
        if encoded_msg[0] != 0x01:
            raise ValueError(
                "The encoded message doesn't start with the correct" " token."
            )
        if encoded_msg[-1] != 0x00:
            raise ValueError(
                "The encoded message doesn't end with the correct" " token."
            )

        encoded_msg_len = int.from_bytes(
            encoded_msg[1:5], byteorder="big", signed=False
        )
        if len(encoded_msg) != encoded_msg_len:
            raise ValueError(
                "The length of the encoded message doesn't match the value in"
                + " its length field."
            )

        # Decode the data
        msg = cls(None)
        msg.id = int.from_bytes(
            encoded_msg[5:21], byteorder="big", signed=False
        )
        msg.payload = json.loads(encoded_msg[21:-1].decode("utf-8"))
        return msg

    def to_bytes(self) -> bytes:
        """Encode the message.

        Returns:
            - _: bytes
                  Encoded message bytes that are ready to be sent.
        """

        msg = json.dumps(self.payload).encode("utf-8")
        # len = payload + tokens + length field + ID field
        msg_len = len(msg) + 2 + 4 + 16

        return (
            b"\x01"
            + msg_len.to_bytes(4, byteorder="big", signed=False)
            + self.id.to_bytes(16, byteorder="big", signed=False)
            + msg
            + b"\x00"
        )
