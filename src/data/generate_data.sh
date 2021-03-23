#!/bin/bash

#SBATCH --job-name=data_generation
#SBATCH -t 2:00:00

ST=$1
MT=$2
MP=$3
PHYS_LEN=$4
N_PER_POP=$5
N_SIMS_PER_JOB=$6
DONOR_POP=$7
ODIR=$8
SCRIPT=$9

python3 src/data/runAndParseSlim.py ${SCRIPT} ${N_SIMS_PER_JOB} ${PHYS_LEN} ${DONOR_POP} ${ODIR}/sim.${SLURM_ARRAY_TASK_ID}.log 1> ${ODIR}/sim.${SLURM_ARRAY_TASK_ID}.ms ${N_PER_POP} ${ST} ${MT} ${MP}

