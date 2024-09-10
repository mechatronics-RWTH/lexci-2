# Installation

This is an extensive guide to installing LExCI on your computer(s). If you only
need a quick summary of the most important commands, check out the
[instructions in the read-me file](https://github.com/mechatronics-RWTH/lexci-2/blob/main/README.md#installation)
of the repository.


## Virtual Environment

Virtual environments are local copies of a Python installation (or parts
thereof) that do not interfere with the one used by the operating system. When
activated in a terminal, they take precedence over the system-wide installation
so that their versions of the language or packages are employed. This effect is
strictly limited to terminals where such an environment has been activated;
other processes are not influenced in any way, shape, or form. As a consequence,
virtual environments allow us to have different versions of Python or its
packages installed on our system at the same time.

LExCI offers little flexibility concerning the versions of its dependencies and
it modifies Ray/RLlib when being set up. Because of that, it is strongly advised
to create a dedicated virtual environment for the framework. Which of the two
solutions presented here you go for is up to your preferences, needs, and
circumstances.


### venv
[venv](https://docs.python.org/3/library/venv.html) is a native Python module
for managing virtual environments. It doesn't take care of handling Python
versions, though. Since that part must be done by hand, this option might be
less appealing to the inexperienced user.


#### Linux
In order to create and activate LExCI's environment, do the following:
01. Install Python 3.9.15:
    01. Download the
        [source code](https://www.python.org/downloads/release/python-3915/) and
        extract the tarball.
    02. As an administrator, install the software packages you'll need for
        building:

        ```
        apt install build-essential gcc gpp g++ gdb libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev tk-dev libffi-dev uuid-dev
        ```

    03. `cd` to the extracted directory containing the source code.
    04. Run these commands as an administrator:
        
        ```
        ./configure --enable-shared
        make -j$(nproc --all)
        make altinstall
        ldconfig
        ```

        Note that `make` is executed with the target `altinstall`. This ensures
        that your base installation is not overwritten.
02. Create a folder for your virtual environment(s) in your home directory by
    typing `mkdir ~/.venv` into a terminal. You can choose a different location
    if you wish. In that case, don't forget to adapt the following commands
    accordingly.
03. Create the environment proper with `python3.9 -m venv ~/.venv/lexci2`.
04. Activate the environment with `source ~/.venv/lexci2/bin/activate`. The
    command line prompt should start with `(lexci2)` now.

The virtual environment can be deactivated by typing `deactivate`. Should you
want to remove it altogether, simply delete `~/.venv/lexci2`. Note that the
folder `~/.venv` is hidden. You may have to change the settings in your file
explorer to make it visible.


#### Windows
In order to create and activate LExCI's environment, do the following:
01. Install Python 3.9.13 by downloading the
    [Windows installer](https://www.python.org/downloads/release/python-3913/)
    and executing it. Select **Install launcher for all users** and click on
    **Install now**.

    Note: LExCI was developed using Python 3.9.15 but — at the time of writing —
    there is no Windows installer available for that version. So, unless you
    feel like building from source, go with the above as the patch version
    number is negligible here.
02. Create a folder for your virtual environment(s) in your home directory by
    typing `mkdir ~/.venv` into a PowerShell. You can choose a different
    location if you wish. In that case, don't forget to adapt the following
    commands accordingly.
03. Create the environment proper with
    `py -3.9 -m venv C:/Users/YOUR_USER_NAME/.venv/lexci2` (`~` must be
    explicitly expanded).
04. Activate the environment with `~/.venv/lexci2/Scripts/Activate.ps1`. You may
    have to
    [change the execution policy of the PowerShell](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.security/set-executionpolicy?view=powershell-7.4)
    for this (e.g. by typing `Set-ExecutionPolicy unrestricted` into a
    PowerShell that you've opened as an administrator). The command line prompt
    should start with `(lexci2)` when the environment is active.

The virtual environment can be deactivated by typing `deactivate`. Should you
want to remove it altogether, simply delete `~/.venv/lexci2`.


### Anaconda
[Anaconda](https://www.anaconda.com/download) is a distribution for managing
environments and packages. If you're not interested in all the extra features it
is shipped with (e.g. a GUI for handling virtual environments or a suite of data
science packages), you can opt for
[Miniconda](https://docs.anaconda.com/miniconda/) which is a lightweight
spin-off containing only the most important elements of the full distribution.
Depending on the size and type of your organisation, you may not meet the
criteria for the free plan, making it paid software. On the plus side, Anaconda
internally takes care of setting up different Python versions for you.

01. Download [Anaconda](https://www.anaconda.com/download) or
    [Miniconda](https://docs.anaconda.com/miniconda/) for your operating system
    and architecture.
02. On Linux, type `bash /path/to/Miniconda3-latest-Linux-x86_64.sh` (replace
    the name of the script with Anaconda's one if you've downloaded that) to
    install the distribution. On Windows, simply double-click on the executable
    and follow the installation wizard.
03. Open a terminal (on Linux) or an Anaconda prompt (on Windows) and type
    `conda create --name lexci2 python=3.9.15` to create the virtual
    environment.
04. Run `conda activate lexci2` to activate the environment. The command line
    prompt should start with `(lexci2)` now.

You can return to the base environment by typing `conda deactivate`. To delete
the whole virtual environment, run `conda env remove --name lexci2`.


## Installing LExCI on Linux

01. Open a terminal and install the required software packages as an
    administrator:
    
    ```
    apt install gcc gpp g++ gdb git
    ```

02. Activate LExCI's virtual environment. See above for the command.
03. Downgrade `pip`, `setuptools`, and `wheel` since newer versions aren't able
    to process the dependencies of the LExCI framework. Also install the version
    of NumPy that LExCI needs.

    ```
    python3.9 -m pip install pip==22.0.4
    python3.9 -m pip install setuptools==58.1.0 wheel==0.38.4 numpy==1.26.4
    ```

04. Navigate to the destination of the repository on your computer and clone it:

    ```
    cd /path/to/local/repo/location
    git clone https://github.com/mechatronics-RWTH/lexci-2.git
    ```

    Then, `cd` into LExCI's folder and check out the version you want to install
    (Here, we're going with `v2.23.0`. Do not use the `main` branch!):

    ```
    cd ./lexci-2
    git checkout v2.23.0
    ```

    Alternatively, you can download the repository as a zip file from GitHub's
    web interface. Make sure to choose the right tag first, though.
05. Type `python3.9 -m pip install .` to start the setup procedure. It takes a
    couple of minutes to complete.
06. If you intend to automate MATLAB/Simulink, type
    `python3.9 -m pip install matlabengine==VERSION` where `VERSION` is the
    latest package version for your MATLAB installation. All available versions
    are listed [here](https://pypi.org/project/matlabengine/#history).

To uninstall LExCI, open a terminal, activate its virtual environment, and type
`pip uninstall lexci-2`.


## Installing LExCI on Windows

One of LExCI's dependencies, [Ray/RLlib](https://github.com/ray-project/ray),
offers limited support for Windows. As a result, only a partial installation of
the framework is possible which, nevertheless, is helpful when writing Minions.

01. Depending on the solution you've chosen for the virtual environment, open a
    PowerShell (when using venv) or an Anaconda prompt.
02. Activate the virtual environment with the respective command above.
03. Downgrade `pip`, `setuptools`, and `wheel` since newer versions aren't able
    to process the dependencies of the LExCI framework. Also install the version
    of NumPy that LExCI needs.

    ```
    py -3.9 -m pip install pip==22.0.4
    py -3.9 -m pip install setuptools==58.1.0 wheel==0.38.4 numpy==1.26.4
    ```

04. Navigate to the destination of the repository on your computer and clone it:

    ```
    cd /path/to/local/repo/location
    git clone https://github.com/mechatronics-RWTH/lexci-2.git
    ```

    Then, `cd` into LExCI's folder and check out the version you want to install
    (Here, we're going with `v2.23.0`. Do not use the `main` branch!):

    ```
    cd ./lexci-2
    git checkout v2.23.0
    ```

    Alternatively, you can download the repository as a zip file from GitHub's
    web interface. Make sure to choose the right tag first, though.
05. Type `py -3.9 -m pip install .` to start the setup procedure.
06. If you intend to automate MATLAB/Simulink, type
    `py -3.9 -m pip install matlabengine==VERSION` where `VERSION` is the latest
    package version for your MATLAB installation. All available versions are
    listed [here](https://pypi.org/project/matlabengine/#history).

To uninstall LExCI, open a terminal, activate its virtual environment, and type
`py -3.9 -m pip uninstall lexci-2`.


## Notes

This how-to was written for the following operating systems:
- Debian GNU/Linux 12 "bookworm"
- Microsoft Windows 10
