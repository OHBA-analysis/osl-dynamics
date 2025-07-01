"""Print the free energy of different runs.

"""

from sys import argv

if len(argv) != 2:
    print("Please pass the number of states, e.g. python 2_print_free_energy.py 8")
    exit()
n_states = int(argv[1])

import pickle
import numpy as np

best_fe = np.Inf
for run in range(1, 11):
    try:
        history = pickle.load(
            open(f"results/{n_states}_states/run{run:02d}/model/history.pkl", "rb")
        )
        free_energy = history["free_energy"]
        print(f"run {run}: {free_energy}")
        if free_energy < best_fe:
            best_run = run
            best_fe = free_energy
    except:
        print(f"run {run} missing")

print()
print("best run:", best_run)
