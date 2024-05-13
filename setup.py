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
        """Perform the installation. If on Linux, this will fist build the
        `nnexec` library, run the standard installation procedure, and finally
        patch RLlib."""

        # If on Linux, build the C library for executing NNs
        is_linux = platform.platform().startswith("Linux")
        if is_linux:
            self._build_libnnexec()

        # Run the standard installation procedure
        super().run()

        # If on Linux, explicitly copy `libnnexec.so` and patch RLlib
        if is_linux:
            self._copy_libnnexec()
            self._patch_rllib()

    def _build_libnnexec(self) -> None:
        """Build the `nnexec` library."""

        path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "lexci2/nnexec"
        )
        cmd = f"cd {path} && make -j`nproc` && make clean_objs"
        subprocess.run(cmd, shell=True, check=True)

    def _copy_libnnexec(self) -> None:
        """Manually copy `libnnexec.so` into its intended destination as
        setuptools fails to do that if the shared library isn't already present
        by the time the script is run. Thus, one doesn't have to run the
        installation twice.
        """

        # Define source and target for copying `libnnexec.so`
        source = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "lexci2/nnexec/libnnexec.so",
        )
        destination = os.path.join(site.getsitepackages()[0], "lexci2/nnexec")

        # Create the destination folder
        pathlib.Path(destination).mkdir(parents=True, exist_ok=True)

        # Copy
        shutil.copy(source, destination)

    def _patch_rllib(self) -> None:
        """Patch RLlib."""

        # Ensure that Ray/RLlib is already installed at this point
        subprocess.run("pip install ray==1.13.0", shell=True, check=True)

        # Get the absolute path to the patch file
        patch_file_name = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "lexci_rllib.patch"
        )
        # Get the absolute path to where Ray/RLlib has been installed
        import ray
        ray_path = os.path.abspath(os.path.dirname(ray.__file__))

        # Apply the patch
        cmd = f"cd {ray_path} && git apply {patch_file_name}"
        subprocess.run(cmd, shell=True, check=False)


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

    # Set platform-specific settings
    platform_str = platform.platform()
    if platform_str.startswith("Linux"):
        required_packages = [
            # ray[all]
            "ray (==1.13.0)",
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
        ]
        package_data = {"lexci2": ["nnexec/libnnexec.so"]}
        entry_points = {
            "console_scripts": [
                "Lexci2UniversalPpoMaster = lexci2.universal_masters.universal_ppo_master.universal_ppo_master:main",
                "Lexci2UniversalDdpgMaster = lexci2.universal_masters.universal_ddpg_master.universal_ddpg_master:main",
            ]
        }
    elif platform_str.startswith("Windows"):
        required_packages = [
            "gym (==0.21.0)",
            "tensorflow (==2.11.0)",
            "pandas (==2.1.1)",
            "dm_tree (==0.1.8)",
            "gputil (==1.4.0)",
            "asammdf (==7.3.14)",
            "pydantic (==1.10.12)",
            "pywin32 (==300.0)",
        ]
        package_data = {}
        entry_points = {}
    else:
        raise RuntimeError(f"Unsupported platform '{platform_str}'.")

    setup(
        name="lexci-2",
        version="2.20.0",
        description="The Learning and Experiencing Cycle Interface (LExCI).",
        author="Kevin Badalian",
        author_email="badalian_k@mmp.rwth-aachen.de",
        url="https://github.com/mechatronics-RWTH/lexci-2",
        packages=find_packages(),
        package_data=package_data,
        include_package_data=True,
        python_requires=">=3.9.15",
        requires=required_packages,
        install_requires=required_packages,
        setup_requires=required_packages,
        entry_points=entry_points,
        cmdclass={"install": LexciInstallationCommand},
    )


if __name__ == "__main__":
    main()
