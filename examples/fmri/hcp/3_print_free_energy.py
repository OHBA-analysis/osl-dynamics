"""Print the free energy of different runs.

"""

import pickle
import numpy as np

best_fe = np.Inf
for i in range(1, 11):
    try:
        history = pickle.load(open(f"results/run{i:02d}/model/history.pkl", "rb"))
        free_energy = history["free_energy"]
        print(f"run {i}: {free_energy}")
        if free_energy < best_fe:
            best_run = i
            best_fe = free_energy
    except:
        print(f"run {i} missing")

print()
print("best run:", best_run)
