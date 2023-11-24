"""Abstract base class for threads that continuously invoke a function.

File: utils/continuous_thread.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date: 2022-05-06


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


import threading
import time
import datetime
import logging
from abc import ABCMeta, abstractmethod


logger = logging.getLogger(__name__)


class ContinuousThread(metaclass=ABCMeta):
    """Thread that continuously executes a function."""

    def __init__(self, period: float = 0.01) -> None:
        """Initialize the continuous thread.

        Arguments:
            - period: float (Default: 0.01, Unit: s)
                  Periodic time of the thread's main loop.
        """

        self._period = period

        self._b_stop = False
        self._lock = threading.Lock()
        self._thread = None

    def start(self) -> None:
        """Start the thread if it isn't already running."""

        if self._thread is None:
            self._thread = threading.Thread(target=self._mainloop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Terminate the continuous thread."""

        self.lock()
        self._b_stop = True
        self.unlock()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def is_running(self) -> bool:
        """Return whether the continuous thread is running.

        Returns:
            - _: bool
                  `True` if the continuous thread is running, else `False`.
        """

        return self._thread is not None and self._thread.is_alive()

    def lock(self) -> None:
        """Lock the thread."""

        self._lock.acquire()

    def unlock(self) -> None:
        """Unlock the thread."""

        self._lock.release()

    @abstractmethod
    def function(self) -> None:
        """Non-blocking function that takes care of its own exception handling
        and is continuously called by the thread.
        """

        raise NotImplementedError

    def _mainloop(self) -> None:
        """Main loop of the tread."""

        while True:
            t_start = datetime.datetime.now()

            # Check if the continuous thread shall be terminated
            self.lock()
            if self._b_stop:
                self.unlock()
                break
            self.unlock()

            # Execute the function
            try:
                self.function()
            except:
                logger.error(
                    "An error occurred in the main loop of the"
                    + " `ContinuousThread`, causing it to stop."
                )
                break

            # Wait for the period to finish
            t_end = datetime.datetime.now()
            t_wait = self._period - (t_end - t_start).total_seconds()
            if t_wait > 0:
                time.sleep(t_wait)
