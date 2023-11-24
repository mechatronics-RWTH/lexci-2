"""Container for shared objects.

File: utils/shared_object.py
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


import threading
from typing import Any


class SharedObject:
    """General lockable container for objects that are meant to be shared
    between threads.
    """

    def __init__(self, obj: Any) -> None:
        """Initialize the shared object.

        Arguments:
            - obj: Any
                  Object to be contained.
        """

        self.obj = obj
        self._lock = threading.Lock()

    def lock(self) -> None:
        """Lock the shared object."""

        self._lock.acquire()

    def unlock(self) -> None:
        """Unlock the shared object."""

        self._lock.release()
