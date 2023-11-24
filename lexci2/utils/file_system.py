"""Helper functions for working with the operating system's file system.

File:   lexci2/utils/file_system.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2022-11-15


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

from typing import Union


def find_newest_folder(root_dir: str) -> Union[str, None]:
    """Find the newest directory in a root folder.

    Arguments:
        - root_dir: str
              Root directory to start the search from.

    Returns:
        - _: Union[str, None]
              Absolute path to the newest directory or `None` if `root_dir`
              doesn't contain any folders.
    """

    dirs = [
        os.path.join(root_dir, e)
        for e in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, e))
    ]
    if len(dirs) == 0:
        return None
    dirs = sorted(dirs, key=lambda x: os.path.getctime(x), reverse=True)
    return dirs[0]


def find_file_in_folder(root_dir: str, file_name: str) -> Union[str, None]:
    """Find a file in a folder or its sub-directories. Please note that this
    function stops once there's a hit. Hence, if there are multiple occurrences
    of the file name, the first one is returned.

    Arguments:
        - root_dir: str
              Root directory to start the search from.
        - file_name: str
              Name of the file to find, including its extension. Make sure that
              the extension's capitalization matches what you're looking for!

    Returns:
        - _: Union[str, None]
              Absolute path to the file with the given name or `None` if it
              isn't there.
    """

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if file_name in filenames:
            return os.path.join(dirpath, file_name)
    return None
