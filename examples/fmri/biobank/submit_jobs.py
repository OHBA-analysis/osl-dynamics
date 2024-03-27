"""Submit jobs to the BMRC cluster.

"""

import os

def write_job_script(run, queue="short", n_gpus=1, n_cpus=12):
    """Create a job script to submit."""

    with open("job.sh", "w") as file:
        name = f"ukb-hmm-{run}"
        file.write("#!/bin/bash\n")
        file.write(f"#SBATCH -J {name}\n")
        file.write(f"#SBATCH -o logs/{name}.out\n")
        file.write(f"#SBATCH -e logs/{name}.err\n")
        file.write(f"#SBATCH -p {queue}\n")
        if "gpu" in queue:
            file.write(f"#SBATCH --gres gpu:{n_gpus}\n")
        else:
            file.write(f"#SBATCH -c {n_cpus}\n")
        file.write("source activate osld\n")
        file.write(f"python 2_train_hmm.py {run}\n")

# Create directory to hold log/error files
os.makedirs("logs", exist_ok=True)

# Submit jobs
for run in range(1, 11):
    write_job_script(run)
    os.system("sbatch job.sh")
    os.system("rm job.sh")
