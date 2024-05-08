# LExCI 2

Version 2 of the **L**earning and **Ex**periencing **C**ycle **I**nterface
(LExCI).


## Installation

You are *highly* encouraged to install LExCI in a virtual environment using
[venv](https://docs.python.org/3/library/venv.html) or
[Anaconda](https://docs.anaconda.com/free/anaconda/)/[Miniconda](https://docs.anaconda.com/free/miniconda/).
That's because i) the setup script demands fixed versions of LExCI's
dependencies and ii) because the `ray` package is patched during installation.

### Linux

1. Install required software packages by typing
   `apt install gcc gpp g++ gdb git` into a terminal.
2. Activate the virtual environment.
3. Run `python3 -m pip pip install pip==22.0.4` and then
   `pip install setuptools==58.1.0 wheel==0.38.4` as newer versions of these
   packages may not be able to install LExCI's dependencies.
4. Download or clone this repository and `cd` to its location.
5. Run `pip install -v .`.


### Windows

LExCI cannot be fully installed on Windows as its `Master` needs
[Ray/RLlib](https://github.com/ray-project/ray) which offers limited support for
the operating system. Nevertheless, a partial setup can be done in order to
facilitate writing Minions.

1. Open a PowerShell or Anaconda prompt.
2. Activate the virtual environment.
3. Run `python.exe -m pip pip install pip==22.0.4` and then
   `pip install setuptools==58.1.0 wheel==0.38.4` as newer versions of these
   packages may not be able to install LExCI's dependencies.
4. Download or clone this repository and `cd` to its location.
5. Run `pip install -v .`.


## Publications

- [Badalian, K., Koch, L., Brinkmann, T., Picerno, M., Wegener, M., Lee, S. Y., & Andert, J. (2023). LExCI: A Framework for Reinforcement Learning with Embedded Systems. arXiv preprint arXiv:2312.02739](https://arxiv.org/pdf/2312.02739)

If you use LExCI in your scientific work, please cite (the pre-print version of)
our LExCI paper:

    @article{badalian2023lexci,
        title={LExCI: A Framework for Reinforcement Learning with Embedded Systems},
        author={Badalian, Kevin and Koch, Lucas and Brinkmann, Tobias and Picerno, Mario and Wegener, Marius and Lee, Sung-Yong and Andert, Jakob},
        journal={arXiv preprint arXiv:2312.02739},
        year={2023},
        url={https://arxiv.org/pdf/2312.02739}
    }


## License

LExCI 2 is licensed under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt).
