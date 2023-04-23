Installation
====================================

Install using pip (developers)
------------------------------
The best way to install mudslide is using pip. First, clone the repository

    git clone github.com/smparker/mudslide.git
    cd mudslide
    pip install --user -e .

This will install the python package and the command line scripts
to your user installation directory.
You can find out your user installation directory with the command

    python -m site --user-base

To set up your `PATH` and `PYTHONPATH` to be able to use both the command line scripts
and the python package, use

    export PATH=$(python -m site --user-base)/bin:$PATH
    export PYTHONPATH=$(python -m site --user-base):$PYTHONPATH
