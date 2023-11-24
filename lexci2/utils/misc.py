"""Miscellaneous helper functions.

File:   lexci2/utils/misc.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2022-08-15


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


import datetime


def get_datetime_str() -> str:
    """Get the current time and date in a format that can be used as a file name.

    Returns:
      - _: str
          Date and time as a string that is compatible with the OS's file naming
          rules.
    """

    s = str(datetime.datetime.now())
    return s.replace(":", "-").replace(" ", "_")
