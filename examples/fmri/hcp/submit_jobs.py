"""Submit jobs to the BMRC cluster.

"""

import os

def write_job_script(run, queue="gpu_short"):
    """Create a job script to submit."""

    with open("job.sh", "w") as file:
        name = f"hcp-hmm-{run}"
        file.write("#!/bin/bash\n")
        file.write(f"#SBATCH -J {name}\n")
        file.write(f"#SBATCH -o logs/{name}.out\n")
        file.write(f"#SBATCH -e logs/{name}.err\n")
        file.write(f"#SBATCH -p {queue}\n")
        file.write("#SBATCH --gres gpu:1\n")
        file.write("source activate osld\n")
        file.write(f"python 2_train_hmm.py {run}\n")

os.makedirs("logs", exist_ok=True)

for run in range(1, 11):
    write_job_script(run)
    os.system("sbatch job.sh")
    os.system("rm job.sh")
