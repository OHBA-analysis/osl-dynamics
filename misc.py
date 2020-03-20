import scipy.io as spio
import numpy as np
from glob import glob
from tqdm import tqdm


def convert_mat_to_npy():
    root = "bSNR_1_iteration_1/"

    file_names = glob(root + '*.mat')

    for mat_file in tqdm(file_names):
        for val in spio.loadmat(mat_file).values():
            if isinstance(val, np.ndarray):
                np.save(mat_file[:-4], val)


def log_print(text, log_type='none'):
    colors = dict(
        magenta='\033[95m',
        blue='\033[94m',
        green='\033[92m',
        yellow='\033[93m',
        fail='\033[91m',
        bold='\033[1m',
        underline='\033[4m',
        red='\033[31m',
        none=''
    )
    end_character = '' if log_type == 'none' else '\033[0m'

    try:
        color = colors[log_type]
    except KeyError:
        color = colors['none']

    print(color + text + end_character)
