#!/bin/bash

# Submit jobs to rusty
sbatch --time=0-00:10 sbatch_rusty.sh 32
sbatch --time=0-00:10 sbatch_rusty.sh 64
sbatch --time=0-00:20 sbatch_rusty.sh 128
sbatch --time=0-01:30 sbatch_rusty.sh 256
sbatch --time=0-08:00 sbatch_rusty.sh 512
sbatch --time=0-20:00 --exclusive --nodes=1 --ntasks-per-node=8 --gpus-per-node=8 --cpus-per-task=8 sbatch_rusty.sh 1024
