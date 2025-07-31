# navier-stokes-turbulence

Philip Mocz (2025)

A simple Navier-Stokes solver in JAX
used the investigate the power spectrum of turbulence


## Virtual Environment

```console
module purge
module load python/3.11
python -m venv --system-site-packages $VENVDIR/navier-stokes-turbulence-venv
source $VENVDIR/navier-stokes-turbulence-venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```


## Run Locally

```console
python navier-stokes-turbulence.py
```


## Submit job (Rusty)

```console
sbatch sbatch_rusty.sh
```

## Analyze results

```console
python analyze.py
```
