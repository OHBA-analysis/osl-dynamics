"""Submit jobs to the BMRC cluster.

"""

import os


def write_gpu_job_script(run, queue="gpu_short"):
    """Create a job script to submit a job to the GPU queue."""

    with open("job.sh", "w") as file:
        name = f"ukb-hmm-{run}"
        file.write("#!/bin/bash\n")
        file.write(f"#SBATCH -J {name}\n")
        file.write(f"#SBATCH -o logs/{name}.out\n")
        file.write(f"#SBATCH -e logs/{name}.err\n")
        file.write(f"#SBATCH -p {queue}\n")
        file.write("#SBATCH --gres gpu:1\n")
        file.write("source activate osld\n")
        file.write(f"python 2_train_hmm.py {run}\n")


def write_cpu_job_script(run, queue="short", n_cpus=24):
    """Create a job script to submit."""

    with open("job.sh", "w") as file:
        name = f"ukb-hmm-{run}"
        file.write("#!/bin/bash\n")
        file.write(f"#SBATCH -J {name}\n")
        file.write(f"#SBATCH -o logs/{name}.out\n")
        file.write(f"#SBATCH -e logs/{name}.err\n")
        file.write(f"#SBATCH -p {queue}\n")
        file.write(f"#SBATCH -c {n_cpus}\n")
        file.write("source activate osld\n")
        file.write(f"python 2_train_hmm.py {run}\n")


os.makedirs("logs", exist_ok=True)

for run in range(1, 11):
    write_cpu_job_script(run)
    os.system("sbatch job.sh")
    os.system("rm job.sh")
