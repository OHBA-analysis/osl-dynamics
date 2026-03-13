"""Submit jobs to the BMRC cluster.

"""

import os

def write_job_script(n_states, run):
    with open("job.sh", "w") as file:
        name = f"cc-{n_states}-{run}"
        file.write("#!/bin/bash\n")
        file.write(f"#SBATCH -J {name}\n")
        file.write(f"#SBATCH -o logs/{name}.out\n")
        file.write(f"#SBATCH -e logs/{name}.err\n")
        file.write("#SBATCH -p gpu_long\n")
        file.write("#SBATCH --gres gpu:1 --constraint 'rtx8000|a100'\n")
        file.write("#SBATCH --mem-per-gpu 200G\n\n")
        file.write("source activate osld\n")
        file.write(f"python 2_train_hmm.py {n_states} {run}\n")

os.makedirs("logs", exist_ok=True)

for n_states in [8, 10, 12]:
    for run in [1, 2, 3, 4, 5]:
        write_job_script(n_states, run)
        os.system("sbatch job.sh")
        os.system("rm job.sh")
