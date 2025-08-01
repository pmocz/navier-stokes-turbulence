#!/bin/bash

# Submit jobs to rusty
sbatch --time=0-00:01:00 sbatch_rusty.sh 64
sbatch --time=0-00:01:00 sbatch_rusty.sh 128
sbatch --time=0-00:01:00 sbatch_rusty.sh 256
sbatch --time=0-00:08:00 sbatch_rusty.sh 512
