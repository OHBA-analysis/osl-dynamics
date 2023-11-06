import random

import pathlib
import numpy as np
import scipy.stats as stats

from osl_dynamics.data import Data

class PrepareData():
    def __init__(self,data_dir:pathlib.Path,n_timepoint:int=1200):
        self.data_dir = data_dir
        self.n_timepoint = n_timepoint
        
    def load(self,split_session:bool = True,split_strategy:str ='0'):
        '''
        Load data from specified directories
        Parameters
        ----------
        split_session: (bool) whether to split the session
        split_strategy: (str) how to split the whole dataset
        '0': no splitting
        '1': randomly split into half
        '2': For each subject, session 12 - 34
        '3': For each subject, session 13 - 24
        '4': For each subject, session 14 - 23

        Returns
        -------
        tuple: A tuple containing the following
            - subjs (list): A list of read-in subjects
            - dataset (osl_dynamics.data.Data): the wrapped dataset
            - dataset_2 (osl_dynamics.data.Data): if split strategy ! '0', the second half needs to be returned.
        '''
        subjs = []
        data_list = []
        for file in sorted(self.data_dir.glob('*.txt')):
            subjs.append(file.stem)
            loaded_data = np.loadtxt(file)
            n_session = len(loaded_data) / self.n_timepoint
            if split_session:
                splitted_data = np.split(loaded_data,n_session)
                for i in range(len(splitted_data)):
                    data_list.append(z_score(splitted_data[i]))
            else:
                data_list.append(z_score(loaded_data,n_session))
        
        print('Read from directory: ',self.data_dir)
        print('Number of subjects: ',len(subjs))

        if split_strategy == '0':
            return subjs, Data(data_list,load_memmaps=False)
        elif split_strategy == '1':
            # Set the random seed for reproducibility
            random_seed = 42
            random.seed(random_seed)
            random_index = random.sample(range(len(data_list)),int(len(data_list) / 2))

            first_half = [data_list[i] for i in random_index]
            second_half = [array for i, array in enumerate(data_list) if i not in random_index]
            return subjs, Data(first_half,load_memmaps=False), Data(second_half,load_memmaps=False)
        elif split_strategy == '2':
            N = len(data_list) // self.n_session
            # Divide the list into groups: 4i, 4i+1 and 4i+2, 4i+3
            first_half = [data_list[self.n_session * i] for i in range(N)]
            first_half.extend(data_list[self.n_session * i + 1] for i in range(N))

            second_half = [data_list[self.n_session * i + 2] for i in range(N)]
            second_half.extend(data_list[self.n_session * i + 3] for i in range(N))
            return subjs, Data(first_half, load_memmaps=False), Data(second_half, load_memmaps=False)
        elif split_strategy == '3':
            N = len(data_list) // self.n_session
            # Divide the list into groups: 4i, 4i+2 and 4i+1, 4i+3
            first_half = [data_list[self.n_session * i] for i in range(N)]
            first_half.extend(data_list[self.n_session * i + 2] for i in range(N))

            second_half = [data_list[self.n_session * i + 1] for i in range(N)]
            second_half.extend(data_list[self.n_session * i + 3] for i in range(N))
            return subjs, Data(first_half, load_memmaps=False), Data(second_half, load_memmaps=False)
        elif split_strategy == '4':
            N = len(data_list) // self.n_session
            # Divide the list into groups: 4i, 4i+3 and 4i+1, 4i+2
            first_half = [data_list[self.n_session * i] for i in range(N)]
            first_half.extend(data_list[self.n_session * i + 3] for i in range(N))

            second_half = [data_list[self.n_session * i + 1] for i in range(N)]
            second_half.extend(data_list[self.n_session * i + 2] for i in range(N))
            return subjs, Data(first_half, load_memmaps=False), Data(second_half, load_memmaps=False)
        else:
            raise ValueError('Incorrect split strategy!')



def z_score(data:np.ndarray,n_session:int = 1):
    """
    z_score the input data.
    If n_session = 1, then z_score directly
    If n_session > 1, then divide the data into different sessions,
    z-score separately, and then concatenate back.
    
    Parameters:
    data (np.ndarray): (n_timepoints,n_channel)
    n_session (int): the number of sessions
    
    Returns
    np.ndarray: z-scored data
    """
    
    if n_session == 1:
        return stats.zscore(data,axis=0)
    else:
        if len(data) % n_session > 0:
            raise ValueError('Number of time points is not divisible by n_session!')
        n_timepoint, n_channel = data.shape
        
        #  Split to n sessions
        data = np.reshape(data,(n_session,-1,n_channel))
        
        # z-score
        data = stats.zscore(data,axis=1)
        
        # Concatenate and reshape
        return np.reshape(data,(n_timepoint,n_channel))
