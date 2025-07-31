#!/bin/bash

# Run the Navier-Stokes simulation
python navier-stokes-turbulence.py --res 32
python navier-stokes-turbulence.py --res 64
python navier-stokes-turbulence.py --res 128

# Analyze the results
python analyze.py --res 32
python analyze.py --res 64
python analyze.py --res 128

# Make summary plot
python summarize.py