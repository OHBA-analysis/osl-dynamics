import pathlib
import numpy as np
import scipy.stats as stats

from osl_dynamics.data import Data

class PrepareData():
    def __init__(self,data_dir:pathlib.Path,n_session:int=4):
        self.data_dir = data_dir
        self.n_session = n_session
        
    def load(self,split_session = True):
        '''
        Load data from specified directories
        Returns:
        tuple: A tuple containing the following
            - subjs (list): A list of read-in subjects
            - dataset (osl_dynamics.data.Data): the wrappeed dataset
        '''
        subjs = []
        data_list = []
        for file in sorted(self.data_dir.glob('*.txt')):
            subjs.append(file.stem)
            loaded_data = np.loadtxt(file)
            if split_session:
                splitted_data = np.split(loaded_data,self.n_session)
                for i in range(len(splitted_data)):
                    data_list.append(z_score(splitted_data[i]))
            else:
                data_list.append(z_score(loaded_data,self.n_session))
        
        print('Read from directory: ',self.data_dir)
        print('Number of subjects: ',len(subjs))
        
        return subjs, Data(data_list)


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