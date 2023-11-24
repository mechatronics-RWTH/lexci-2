"""Class for inter-process communication via named pipes.

File:   lexci2/utils/pipe_com.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2023-04-20


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


import os
import threading
import select
import time
import logging


logger = logging.getLogger(__name__)


class _PipeReader:
    """Helper class for reading from a named pipe."""

    def __init__(self, pipe_name: str, b_pipe_owner: bool) -> None:
        """Initialize the reader.

        Arguments:
            - pipe_name: str
                  Name of the pipe to read from.
            - b_pipe_owner: bool
                  Whether this object owns the named pipe and is therefore
                  responsible for its creation and deletion.
        """

        # Copy parameters
        self._pipe_name = pipe_name
        self._b_pipe_owner = b_pipe_owner

        # Initialize attributes
        self._buffer = ""
        self._lock = threading.Lock()
        self._anon_read_pipe, self._anon_write_pipe = os.pipe()

        # Create the named pipe if this object is its owner
        if self._b_pipe_owner:
            try:
                logger.info(f"Creating the named pipe '{self._pipe_name}'.")
                os.mkfifo(self._pipe_name)
            except FileExistsError:
                logger.warn(
                    f"The named pipe '{self._pipe_name}' already exists."
                )

        # Create and start the reading thread
        self._reading_thread = threading.Thread(
            target=self._reading_proc, daemon=True
        )
        self._reading_thread.start()

    def terminate(self) -> None:
        """Terminate the reader."""

        # Stop the reading thread
        if self._reading_thread is not None:
            os.write(self._anon_write_pipe, b"0")  # Content irrelevant
            self._reading_thread.join()
            self._reading_thread = None

        # Close the anonymous pipes
        os.close(self._anon_read_pipe)
        os.close(self._anon_write_pipe)

        # Remove the named pipe if this object is its owner
        if self._b_pipe_owner:
            logger.info(f"Removing the named pipe '{self._pipe_name}'.")
            os.unlink(self._pipe_name)

    def _reading_proc(self) -> None:
        """Reading procedure that is run in a separate thread."""

        while True:
            with open(self._pipe_name, "r") as pipe:
                pipe_selection = select.select(
                    [self._anon_read_pipe, pipe], [], []
                )
                if self._anon_read_pipe in pipe_selection[0]:
                    break
                elif pipe in pipe_selection[0]:
                    data = pipe.read()
                    self._lock.acquire()
                    self._buffer += data
                    self._lock.release()

    def read(self) -> str:
        """Get the content of the read buffer.

        Returns:
            - _: str
                  Content of the read buffer.
        """

        self._lock.acquire()
        temp = self._buffer
        self._buffer = ""
        self._lock.release()
        return temp


class _PipeWriter:
    """Helper class for writing to an existing named pipe."""

    def __init__(
        self, pipe_name: str, b_pipe_owner: bool, period: float = 0.01
    ) -> None:
        """Initialize the writer.

        Arguments:
            - pipe_name: str
                  Name of the pipe to write to.
            - b_pipe_owner: bool
                  Whether this object owns the named pipe and is therefore
                  responsible for its creation and deletion.
            - period: float (Default: 0.01, Unit: s)
                  Periodic time of the writing thread.
        """

        # Copy parameters
        self._pipe_name = pipe_name
        self._b_pipe_owner = b_pipe_owner
        self._period = period

        # Initialize attributes
        self._buffer = ""
        self._b_terminate = False
        self._lock = threading.Lock()

        # Create the named buffer if the object is its owner
        if self._b_pipe_owner:
            try:
                logger.info(f"Creating the named pipe '{self._pipe_name}'.")
                os.mkfifo(self._pipe_name)
            except FileExistsError:
                logger.warn(
                    f"The named pipe '{self._pipe_name}' already exists."
                )

        # Create and start the writing thread
        self._writing_thread = threading.Thread(
            target=self._writing_proc, daemon=True
        )
        self._writing_thread.start()

    def terminate(self) -> None:
        """Terminate the writer."""

        # Terminate the writing thread
        if self._writing_thread is not None:
            self._lock.acquire()
            self._b_terminate = True
            self._lock.release()
            self._writing_thread.join()
            self._writing_thread = None

        # Delete the named pipe if this object owns it
        if self._b_pipe_owner:
            logger.info(f"Removing the named pipe '{self._pipe_name}'.")
            os._unlink(self._pipe_name)

    def _writing_proc(self) -> None:
        """Writing procedure that is run in a separate thread."""

        while True:
            # Get the content of the buffer and the termination flag
            self._lock.acquire()
            data = self._buffer
            self._buffer = ""
            b_terminate = self._b_terminate
            self._lock.release()

            # Stop the thread if the writer has been terminated
            if b_terminate:
                break

            if len(data) > 0:
                with open(self._pipe_name, "w") as pipe:
                    # Send the data
                    pipe.write(data)
                    pipe.flush()

            time.sleep(self._period)

    def write(self, data: str) -> None:
        """Write data to the named pipe.

        Arguments:
            - data: str
                  The data to write.
        """

        self._lock.acquire()
        self._buffer += data
        self._lock.release()


class PipeCom:
    """Class for inter-process communication via named pipes."""

    def __init__(self) -> None:
        """Initialize the communicator.

        Use the methods `read_from()` and `write_to()` to define the pipes that
        are used for communication.
        """

        self._reader = None
        self._writer = None

    def __del__(self) -> None:
        """Destroy the object."""

        self.terminate()

    def terminate(self) -> None:
        """Terminate the communicator."""

        if self._reader is not None:
            self._reader.terminate()
            self_reader = None
        if self._writer is not None:
            self._writer.terminate()
            self._writer = None

    def read_from(self, pipe_name: str, b_owns_pipe: bool = False) -> None:
        """Tell the communicator which pipe to read from.

        Arguments:
            - pipe_name: str
                  Name of the pipe to read from.
            - b_owns_pipe: bool (Default: False)
                  Whether the communicator owns the named pipe and therefore
                  takes care of its creation and deletion.
        """

        if self._reader is not None:
            self._reader.terminate()
            self._reader = None

        self._reader = _PipeReader(pipe_name, b_owns_pipe)

    def write_to(self, pipe_name: str, b_owns_pipe: bool = False) -> None:
        """Tell the communicator which pipe to write to.

        Arguments:
            - pipe_name: str
                  Name of the pipe to write to.
            - b_owns_pipe: bool (Default: False)
                  Whether the communicator owns the named pipe and therefore
                  takes care of its creation and deletion.
        """

        if self._writer is not None:
            self._writer.terminate()
            self._writer = None

        self._writer = _PipeWriter(pipe_name, b_owns_pipe)

    def read(self) -> str:
        """Read data from the reading pipe.

        Returns:
            - _: str
                  The data that has been read. If nothing new has been received,
                  this method returns an empty string.

        Raises:
            - A `RuntimeError` is raised if the communicator has no reader.
        """

        if self._reader is None:
            raise RuntimeError("`PipeCom` object has no reader.")

        return self._reader.read()

    def write(self, data: str) -> None:
        """Write data to the writing pipe.

        Arguments:
            - data: str
                  The data to write.

        Raises:
          - A `RuntimeError` is raised if the communicator has no writer.
        """

        if self._writer is None:
            raise RuntimeError("`PipeCom` object has no writer.")

        self._writer.write(data)
