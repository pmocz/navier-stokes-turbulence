#!/bin/bash

# Create virtual environment on rusty
rm -fr $VENVDIR/navier-stokes-turbulence-venv

module purge
module load python/3.13
python -m venv --system-site-packages $VENVDIR/navier-stokes-turbulence-venv
source $VENVDIR/navier-stokes-turbulence-venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
