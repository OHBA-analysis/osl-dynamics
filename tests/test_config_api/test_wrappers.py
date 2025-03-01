import numpy as np
import numpy.testing as npt
from osl_dynamics.config_api.wrappers import load_data


def test_load_data():
    import os
    import json
    save_dir = './test_temp/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    vector = np.array([-1.5 ** 0.5, 0, 1.5 ** 0.5])
    input_1 = np.array([vector, vector + 10.0]).T
    input_2 = np.array([vector * 0.5 + 1., vector * 100]).T
    input_3 = np.zeros((3,2))
    np.savetxt(f'{save_dir}10001.txt', input_1)
    np.savetxt(f'{save_dir}10002.txt', input_2)
    np.savetxt(f'{save_dir}10003.txt',input_3)
    prepare = {'select':{'sessions':[0,1]},'standardize': {}}

    data = load_data(inputs=save_dir, prepare=prepare)
    npt.assert_equal(data.n_sessions,2)
    npt.assert_almost_equal(data.arrays[0], np.array([vector, vector]).T)
    npt.assert_almost_equal(data.arrays[1], np.array([vector, vector]).T)
    data.delete_dir()
    # Remove the directory after testing
    from shutil import rmtree
    rmtree(save_dir)