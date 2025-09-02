#!/bin/bash

# Submit jobs to rusty
sbatch --time=0-00:10 --job-name=ns32 sbatch_rusty.sh 32
sbatch --time=0-00:10 --job-name=ns64 sbatch_rusty.sh 64
sbatch --time=0-00:20 --job-name=ns128 sbatch_rusty.sh 128
sbatch --time=0-01:30 --job-name=ns256 sbatch_rusty.sh 256
sbatch --time=0-08:00 --job-name=ns512 sbatch_rusty.sh 512
sbatch --time=1-08:00 --exclusive --nodes=1 --ntasks-per-node=8 --gpus-per-node=8 --cpus-per-task=8 --job-name=ns1024 sbatch_rusty.sh 1024
