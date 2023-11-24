#!/bin/bash
#
# Shell script for installing LExCI 2.
#
# File:   install.sh
# Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
#         Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
#         RWTH Aachen University
# Date:   2022-11-04
#
#
# Copyright 2023 Teaching and Research Area Mechatronics in Mobile Propulsion,
#                RWTH Aachen University
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at: http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#


# Define location variables
INSTALLER_DIR=$(dirname $(realpath $0))
LEXCI_RAY_DIR="$INSTALLER_DIR/external/ray-ray-1.13.0"
VENV_DIR="$HOME/.venv"
VENV_RAY_DIR="$VENV_DIR/lexci2/lib/python3.9/site-packages/ray"
PYTHON_SRC_URL="https://www.python.org/ftp/python/3.9.15/Python-3.9.15.tar.xz"
PYTHON_SRC_NAME="$(basename $PYTHON_SRC_URL)"


# Install all necessary packages
echo "Installing required packages..."
sudo apt install wget build-essential libssl-dev zlib1g-dev libncurses5-dev\
    libncursesw5-dev libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev\
    libbz2-dev libexpat1-dev liblzma-dev tk-dev libffi-dev uuid-dev python3 pip


# Install Python 3.9.15
BUILD_PYTHON=1
# Check whether the required Python version is already installed
if [ "$(python3 --version)" == "Python 3.9.15" ]; then
  echo "It looks like the OS already has Python 3.9.15. Do you wish to"\
      " re-install? [y/n]"
  read -n 1 -s
  if [ "$REPLY" == "n" ]; then
    BUILD_PYTHON=0
  fi
fi
# Download, build, and install Python
if [ $BUILD_PYTHON -eq 1 ]; then
  echo "Downloading and installing Python 3.9.15..."

  # Create a temporary folder for downloading and building Python
  if [ -d $INSTALLER_DIR/tmp ]; then
    rm -r $INSTALLER_DIR/tmp
  fi
  mkdir $INSTALLER_DIR/tmp

  # Get and build Python
  wget --show-progress -P $INSTALLER_DIR/tmp $PYTHON_SRC_URL
  cd $INSTALLER_DIR/tmp
  tar -xf $PYTHON_SRC_NAME
  cd ${PYTHON_SRC_NAME%.tar.xz}
  ./configure
  make -j$(nproc --all)
  sudo make altinstall
  cd $INSTALLER_DIR

  # Remove the temporary folder
  sudo rm -r $INSTALLER_DIR/tmp
fi


# Check if the venv-directory or the LExCI 2 environment already exist
if [ -d $VENV_DIR ]; then
  if [ -d $VENV_DIR/lexci2 ]; then
    echo "It seems like you've already installed the LExCI 2 environment. Do"\
        " you wish to re-install? [y/n]"
    read -n 1 -s
    if [ "$REPLY" == "y" ]; then
      rm -r $VENV_DIR/lexci2
    else
      exit 0
    fi
  fi
else
  echo "Creating the folder \'$VENV_DIR\'..."
  mkdir $VENV_DIR
fi


# Create the environment
echo "Creating the virtual environment for LExCI 2..."
python3.9 -m venv $VENV_DIR/lexci2
source $VENV_DIR/lexci2/bin/activate
pip install ray[all]==1.13.0 gym==0.21.0 tensorflow==2.11.0 pandas==2.1.1 dm_tree==0.1.8\
    gputil==1.4.0 asammdf==7.3.14 pydantic==1.10.12


# Patch Ray/RLlib
echo "Patching Ray/RLlib..."
cp $LEXCI_RAY_DIR/rllib/agents/ddpg/noop_model.py \
    $VENV_RAY_DIR/rllib/agents/ddpg/
cp $LEXCI_RAY_DIR/rllib/agents/ddpg/lexci_ddpg.py \
    $VENV_RAY_DIR/rllib/agents/ddpg


# Build libnnexec
echo "Building libnnexec..."
cd $INSTALLER_DIR/tools/nnexec/
cp -r $INSTALLER_DIR/external/Simulink/ReinforcementLearningBlock/external_cpp_libs/* ./
make clean
make -j$(nproc --all)
cp ./libnnexec.so $INSTALLER_DIR/lexci2/utils
cd $INSTALLER_DIR


# Clean up
echo "Cleaning up..."
if [ -d $INSTALLER_DIR/tmp ]; then
  rm -r $INSTALLER_DIR/tmp
fi


echo "Installation finished. Type \`source $VENV_DIR/lexci2/bin/activate\` to"\
    " enable LExCI 2's environment or \`deactivate\` to disable it."
exit 0

