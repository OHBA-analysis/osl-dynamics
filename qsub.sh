#!/bin/bash

# Specify a job name
#$ -N swimming

# --- Parameters for the Queue Master ---
# target queue (Please specify in the command line!)
##$ -q short.q

# Run the job in the current working directory
#$ -cwd -j y

# Log locations which are relative to the current
# working directory of the submission
#$ -o ./log/
#$ -e ./log/

# Parallel environemnt settings
#  For more information on these please see the wiki
#  Allowed settings:
#   shmem
#   mpi
#   node_mpi
#   ramdisk
##$ -pe shmem 1

# Print some useful data about the job to help with debugging
echo "------------------------------------------------"
echo "SGE Job ID: $JOB_ID"
echo "SGE Job ID: $SGE_JOB_ID"
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "------------------------------------------------"

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/fmrib/conda/python3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/fmrib/conda/python3/etc/profile.d/conda.sh" ]; then
        . "/opt/fmrib/conda/python3/etc/profile.d/conda.sh"
    else
        export PATH="/opt/fmrib/conda/python3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
# Finally, we can run our real computing job

conda activate osld # Environment name
python post_analysis.py
# End of job script
