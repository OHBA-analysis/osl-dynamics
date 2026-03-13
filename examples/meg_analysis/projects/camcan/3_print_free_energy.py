"""Print free energy.

"""

from sys import argv

if len(argv) != 2:
    print("Please pass the number of states, e.g. python 3_print_free_energy.py 8")
    exit()

n_states = int(argv[1])

output_dir = f"results/models/{n_states:02d}_states"

import pickle
import numpy as np

def get_best_run(min_, max_):
    best_fe = np.Inf
    for run in range(min_, max_ + 1):
        try:
            with open(f"{output_dir}/run{run:02d}/loss.dat") as file:
                lines = file.readlines()
            fe = float(lines[1].split("=")[-1].strip())
            #print(f"run {run}: {fe}")
            if fe < best_fe:
                best_run = run
                best_fe = fe
        except:
            print(f"run {run} missing")
            pass
    return best_run

print("Best run:", get_best_run(1, 5))
