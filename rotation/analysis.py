import os
import pickle

def HMM_analysis(dataset, save_dir):
    from osl_dynamics.models import load
    model = load(save_dir)
    if not os.path.isfile(f'{save_dir}alpha.pkl'):
        alpha = model.get_alpha(dataset)
        pickle.dump(alpha, open(f'{save_dir}alpha.pkl', "wb"))
        
    

def Dynemo_analysis(dataset, save_dir):
    from osl_dynamics.models import load
    model = load(save_dir)
    if not os.path.isfile(f'{save_dir}alpha.pkl'):
        alpha = model.get_alpha(dataset)
        pickle.dump(alpha, open(f'{save_dir}alpha.pkl', "wb"))