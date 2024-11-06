"""Installation script for LExCI 2.

File:   setup.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2024-05-07


Copyright 2024 Teaching and Research Area Mechatronics in Mobile Propulsion,
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
import copy
import platform
import site
import subprocess
import shutil
import pathlib
import logging

from distutils.core import setup
from setuptools import find_packages
from setuptools.command.install import install


# Create the logger
logger = logging.getLogger(__name__)


class LexciInstallationCommand(install):
    """Class containing a custom installation command."""

    def run(self) -> None:
        """Build the `nnexec` library, run the standard installation procedure,
        and finally patch RLlib (if on Linux).
        """

        # Build the library (if possible) and run the standard installation
        # procedure
        if LexciInstallationCommand.is_libnnexec_buildable():
            self._build_libnnexec()
        super().run()
        if LexciInstallationCommand.is_libnnexec_buildable():
            self._copy_libnnexec()

        # On Linux, patch RLlib
        if platform.platform().startswith("Linux"):
            self._patch_rllib()

    @staticmethod
    def is_libnnexec_buildable() -> bool:
        """Check whether the `nnexec` library can be built.

        Returns:
            - _: bool
                  `True` if `nnexec` can be built on the platform, else `False`.
        """

        platform_str = platform.platform()
        if platform_str.startswith("Linux"):
            return True
        elif platform_str.startswith("Windows"):
            if "CYGWIN_PATH" in os.environ:
                return True
            else:
                logger.warning(
                    "The environment variable `CYGWIN_PATH` isn't set, so the"
                    + " `nnexec` library will not be built. This means that you"
                    + " won't be able to directly execute agents on this"
                    + " system. Simply ignore this message if you don't need"
                    + " that feature."
                )
                return False
        else:
            logger.warning(
                "Unsupported platform '{platform_str}'. The `nnexec` library"
                + " will not be built."
            )
            return False

    def _build_libnnexec(self) -> None:
        """Build the `nnexec` library.

        Raises:
            - RuntimeError:
                  - If the system isn't Linux or Windows.
                  - If the environment variable `CYGWIN_PATH` isn't set on
                    Windows.
                  - If the environment variable `PATH` isn't set on Windows.
                    This shouldn't happen under normal circumstances as it is
                    used by the operating system.
        """

        # The working directory of the build process
        path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "lexci2", "nnexec"
        )

        # Set the platform-specific command
        env = copy.deepcopy(os.environ)
        platform_str = platform.platform()
        if platform_str.startswith("Linux"):
            cmd = f"make libnnexec -j`nproc` && make clean_objs"
        elif platform_str.startswith("Windows"):
            # Ensure that the environment variable with the path to Cygwin has
            # been set
            if "CYGWIN_PATH" not in os.environ:
                raise RuntimeError(
                    "The environment variable `CYGWIN_PATH` has not been set."
                )
            if "PATH" not in os.environ:
                raise RuntimeError(
                    "The environment variable `PATH` has not been saved."
                )

            # Add Cygwin's binaries to `PATH`
            cygwin_bin_path = os.path.join(os.environ["CYGWIN_PATH"], "bin")
            env["PATH"] = cygwin_bin_path + os.pathsep + os.environ["PATH"]

            # Create the command string
            cmd = "bash -c 'make libnnexec_win -j`nproc` && make clean_objs'"
        else:
            raise RuntimeError(f"Unsupported platform '{platform_str}'.")

        # Run the process
        subprocess.run(cmd, shell=True, check=True, cwd=path, env=env)

    def _copy_libnnexec(self) -> None:
        """Manually copy `libnnexec.so`/`nnexec.dll` into its intended
        destination as setuptools fails to do that if the shared library isn't
        already present by the time the script is run. Thus, one doesn't have to
        run the installation twice.

        Raises:
            - RuntimeError:
                  - If the system isn't Linux or Windows.
        """

        # Define source and target for copying `libnnexec.so`/`nnexec.dll`
        platform_str = platform.platform()
        if platform_str.startswith("Linux"):
            lib_name = "libnnexec.so"
        elif platform_str.startswith("Windows"):
            lib_name = "nnexec.dll"
        else:
            raise RuntimeError(f"Unsupported platform '{platform_str}'.")
        source = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "lexci2",
            "nnexec",
            lib_name,
        )
        destination = os.path.join(
            site.getsitepackages()[0], "lexci2", "nnexec"
        )

        # Create the destination folder
        pathlib.Path(destination).mkdir(parents=True, exist_ok=True)

        # Copy
        shutil.copy(source, destination)

    def _patch_rllib(self) -> None:
        """Patch RLlib."""

        # Ensure that Ray/RLlib is already installed at this point
        subprocess.run(["pip", "install", "ray==1.13.0"], check=True)

        # Get the absolute path to the patch file
        patch_file_name = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "lexci_rllib.patch"
        )
        # Get the absolute path to where Ray/RLlib has been installed
        import ray

        ray_path = os.path.abspath(os.path.dirname(ray.__file__))

        # Apply the patch
        cmd = f"git apply {patch_file_name}"
        subprocess.run(cmd, shell=True, check=False, cwd=ray_path)


def main() -> None:
    """Main function of the installtion script."""

    # Check if the package is being installed in a virtual environment
    if (
        os.getenv("VIRTUAL_ENV", "") == ""
        and os.getenv("CONDA_DEFAULT_ENV", "") == "base"
    ):
        logger.warning(
            "It seems like you're installing LExCI in your base Python"
            + " environment. You're HIGHLY encouraged to use a virtual"
            + " environment or an Anaconda environment for that."
        )

    required_packages = [
        # ray[all]
        "ray (==1.13.0)",
        "click (==8.0.4)",
        "grpcio (==1.43.0)",
        "pandas (==2.1.1)",
        "pyarrow (==6.0.1)",
        "fsspec (==2024.3.1)",
        "aiohttp (==3.9.5)",
        "aiohttp_cors (==0.7.0)",
        "colorful (==0.5.6)",
        "py_spy (==0.3.14)",
        "requests (==2.31.0)",
        "gpustat (==1.1.1)",
        "opencensus (==0.11.4)",
        "prometheus_client (==0.13.1)",
        "smart_open (==7.0.4)",
        "uvicorn (==0.16.0)",
        "starlette (==0.37.2)",
        "fastapi (==0.111.0)",
        "aiorwlock (==1.4.0)",
        "tabulate (==0.9.0)",
        "tensorboardX (==2.6)",
        "kubernetes (==29.0.0)",
        "urllib3 (==2.2.1)",
        "opentelemetry_api (==1.1.0)",
        "opentelemetry_sdk (==1.1.0)",
        "opentelemetry_exporter_otlp (==1.1.0)",
        "numpy (==1.26.4)",
        "ray_cpp (==1.13.0)",
        "kopf (==1.37.2)",
        "dm_tree (==0.1.8)",
        "gym (==0.21.0)",
        "lz4 (==4.3.3)",
        "matplotlib (==3.8.4)",
        "scikit_image (==0.22.0)",
        "pyyaml (==6.0.1)",
        "scipy (==1.13.0)",
        # Other dependencies
        "tensorflow (==2.11.0)",
        "gputil (==1.4.0)",
        "asammdf (==7.3.14)",
        "pydantic (==1.10.12)",
        "psutil (==6.0.0)",
        "black (==24.8.0)",
    ]

    # Set platform-specific settings
    platform_str = platform.platform()
    if platform_str.startswith("Linux"):
        # Remove `ray` here because it is installed in
        # `LexciInstallationCommand._patch_rllib()`
        required_packages.remove("ray (==1.13.0)")
        package_data = {"lexci2": [os.path.join("nnexec", "libnnexec.so")]}
        entry_points = {
            "console_scripts": [
                "Lexci2UniversalPpoMaster = lexci2.universal_masters.universal_ppo_master.universal_ppo_master:main",
                "Lexci2UniversalDdpgMaster = lexci2.universal_masters.universal_ddpg_master.universal_ddpg_master:main",
                "Lexci2UniversalTd3Master = lexci2.universal_masters.universal_td3_master.universal_td3_master:main",
            ]
        }
    elif platform_str.startswith("Windows"):
        package_data = {}
        if LexciInstallationCommand.is_libnnexec_buildable():
            package_data["lexci2"] = [os.path.join("nnexec", "nnexec.dll")]
        entry_points = {}
    else:
        raise RuntimeError(f"Unsupported platform '{platform_str}'.")

    setup(
        name="lexci-2",
        version="2.23.0",
        description="The Learning and Experiencing Cycle Interface (LExCI).",
        author="Kevin Badalian",
        author_email="badalian_k@mmp.rwth-aachen.de",
        url="https://github.com/mechatronics-RWTH/lexci-2",
        license="Apache-2.0",
        packages=find_packages(exclude=["tests*"]),
        package_data=package_data,
        include_package_data=True,
        # LExCI was developed using Python 3.9.15 but the closest available
        # Windows installer is version 3.9.13
        python_requires=">=3.9.13",
        requires=required_packages,
        install_requires=required_packages,
        setup_requires=required_packages,
        entry_points=entry_points,
        cmdclass={"install": LexciInstallationCommand},
    )


if __name__ == "__main__":
    main()
