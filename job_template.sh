#!/bin/bash

# Set number of nodes and tasks per node
#SBATCH --ntasks-per-node=%cpus%
#SBATCH --nodes=1

# Set max wallclock time
#SBATCH --time=%wall_time%

# Use partition defq|intel (the latter the includes former hydra nodes)
#SBATCH -p milan

# Set account for accessing a specific cpu time contingent
# Options: default, fwd
#SBATCH -A default

# Set name of the job
#SBATCH -J %job_name%

# Mail alert at BEGIN|END|FAIL|ALL
#SBATCH --mail-type=END

# Send mail to the following address
#SBATCH --mail-user=d.fischer@hzdr.de

rm job.log

LOG_FILE=./job.log

function Log {
    local level=$1
    local msg=$2
    echo $(date --rfc-3339=seconds):${level} ${msg} >> ${LOG_FILE}
}

Log INFO "JOB START"
Log INFO "JOB NAME = ${SLURM_JOB_NAME}"

Log INFO "loading modules"
Log INFO "Loading module python ..."
module load python/3.10.4 >> ${LOG_FILE} 2>&1

# Change to execution directory
cd $SLURM_SUBMIT_DIR

# Fix distribution of tasks to nodes as workaround for bug in slurm
# Proposed by Henrik Schulz (FWCI)
export SLURM_TASKS_PER_NODE="$((SLURM_NTASKS / SLURM_NNODES))$( for ((i=2; i<=$SLURM_NNODES; i++)); \
do printf ",$((SLURM_NTASKS / SLURM_NNODES))"; done )"

NODES=$(scontrol show hostname $SLURM_JOB_NODELIST | paste -d, -s)
Log INFO "allocated nodes ${NODES}"
Log INFO "SLURM_NTASKS = ${SLURM_NTASKS}"
rm *.out

python -m venv test-env >> ${LOG_FILE} 2>&1
source test-env/bin/activate >> ${LOG_FILE} 2>&1
python -m pip install --upgrade pip >> $LOG_FILE 2>&1
python -m pip install -r ./requirements.txt >> ${LOG_FILE} 2>&1

# calculating region growing method
python ./methods.py calc_case_ratio >> $LOG_FILE 2>&1

# calculating using intensity window
# python ./fingers.py proc_cases >> $LOG_FILE 2>&1

Log INFO "JOB FINISH"