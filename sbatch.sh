#!/bin/bash 

# Specify a job name 
#SBATCH -J swimming

# Account name and target partition 
#SBATCH -A win.prj 
#SBATCH -p short

# Check whether the directory log is available. Create otherwise.
#if [ ! -d "log" ]; then
#    mkdir log
#fi

# Generate timestamp
#timestamp=$(date +"%Y%m%d_%H%M%S")

# Log locations which are relative to the current 
# working directory of the submission   
#SBATCH --output=./log/slurm_%A_%a.out
 

# Parallel environment settings 
#  For more information on these please see the documentation 
#  Allowed parameters: 
#   -c, --cpus-per-task 
#   -N, --nodes 
#   -n, --ntasks 
#SBATCH -c 1 

# Some useful data about the job to help with debugging 
echo "------------------------------------------------" 
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Slurm Array Task ID: $SLURM_ARRAY_TASK_ID" 
echo "Run on host: "`hostname` 
echo "Operating system: "`uname -s` 
echo "Username: "`whoami` 
echo "Started at: "`date` 
echo "------------------------------------------------" 

# Load the environment
module purge
module load Anaconda3/2022.05
eval "$(conda shell.bash hook)"
conda activate osld # Environment name

# Begin writing your script here 

python main.py $SLURM_ARRAY_TASK_ID

# End of job script
