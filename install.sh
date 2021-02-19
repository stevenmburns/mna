#!/bin/bash
export VENV=`pwd`/pyenv
python3 -m venv $VENV
source $VENV/bin/activate
python install --upgrade pip
pip install wheel
pip install numpy
pip install pytest
pip install cvxpy