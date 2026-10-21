"""Submit the comparison to the BMRC cluster.

One job per (method, selection criterion, run).
"""

import os

methods = ["random_state_time_course", "random_subset", "random_windows", "kmeans"]
selects = ["loss", "free_energy"]
runs = range(1, 11)


def write_job_script(method, select, run):
    with open("job.sh", "w") as file:
        name = f"init-{method}-{select}-{run}"
        file.write("#!/bin/bash\n")
        file.write(f"#SBATCH -J {name}\n")
        file.write(f"#SBATCH -o logs/{name}.out\n")
        file.write(f"#SBATCH -e logs/{name}.err\n")
        file.write("#SBATCH -p gpu_long\n")
        file.write("#SBATCH --gres gpu:1 --constraint 'rtx8000|a100'\n")
        file.write("#SBATCH --mem-per-gpu 200G\n\n")
        file.write("source activate osld\n")
        file.write(f"python 1_train_hmm.py {method} {select} {run}\n")


os.makedirs("logs", exist_ok=True)

for method in methods:
    for select in selects:
        for run in runs:
            write_job_script(method, select, run)
            os.system("sbatch job.sh")
            os.system("rm job.sh")
