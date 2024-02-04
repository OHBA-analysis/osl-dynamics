import numpy as np
import numpy.testing as npt
from osl_dynamics.data import Data

def test_data_nparray():
    # Note: the input should be (n_timepoints,n_channels)
    input = np.array([[1.,0.5],[2.,0.4],[1.5,0.3]])

    data = Data(input)

    # Test self.__iter__
    for obj in data:
        npt.assert_almost_equal(obj,input)
    # Test self.__item__
    npt.assert_almost_equal(data.arrays[0],input)
    # Test @property self.raw_data
    npt.assert_almost_equal(data.raw_data[0],input)
    # Test @property self.n_channels
    npt.assert_equal(data.n_channels,2)
    # Test @property self.n_sammples
    npt.assert_equal(data.n_samples,3)
    # Test set.sampling_frequency
    data.set_sampling_frequency(1.0)
    npt.assert_equal(data.sampling_frequency,1.0)
    # Test set buffer_size
    data.set_buffer_size(100)
    npt.assert_equal(data.buffer_size,100)
    # Test self.select
    data.select(channels=[1],timepoints=[1,3])
    npt.assert_almost_equal(data.arrays[0],input[1:3,1:])

def test_data_list_nparray():
    input_1 = np.array([[1.,0.5],[2.,0.4],[1.5,0.3]])
    input_2 = input_1 / 2
    data = Data([input_1,input_2])
    # Test self.__getitem__
    npt.assert_almost_equal(data.arrays[0],input_1)
    npt.assert_almost_equal(data.arrays[1],input_2)
    # Test @property self.n_arrays
    npt.assert_equal(data.n_arrays,2)

    # Test data.set_keep
    with data.set_keep([0]):
        npt.assert_equal(data.keep,[0])
        for element in data.dataset(sequence_length=3,batch_size=1):
            npt.assert_almost_equal(element['data'].numpy(),np.array([input_1]))

    with data.set_keep([1]):
        npt.assert_equal(data.keep,[1])
        for element in data.dataset(sequence_length=3,batch_size=1):
            npt.assert_almost_equal(element['data'].numpy(),np.array([input_2]))



def test_data_files():
    import os
    import numpy as np
    save_dir = './test_temp/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    input_1 = np.array([[1., 0.5], [2., 0.4], [1.5, 0.3]])
    input_2 = input_1 / 2
    np.save(f'{save_dir}10001.npy',input_1)
    np.savetxt(f'{save_dir}10002.txt',input_2)
    data = Data(save_dir)
    npt.assert_almost_equal(data.arrays,[input_1,input_2])

    # Remove the directory after testing
    from shutil import rmtree
    rmtree(save_dir)

def test_standardize():
    import numpy as np
    vector = np.array([-1.5 ** 0.5,0,1.5 ** 0.5])
    input_1 = np.array([vector,vector + 10.0]).T
    input_2 = np.array([vector * 0.5 + 1.,vector * 100]).T
    data = Data([input_1,input_2])
    data.prepare({'standardize':{}})

    npt.assert_almost_equal(data.arrays[0],np.array([vector,vector]).T)
    npt.assert_almost_equal(data.arrays[1],np.array([vector, vector]).T)
    npt.assert_almost_equal(data.raw_data[0],input_1,decimal=6)
    npt.assert_almost_equal(data.raw_data[1], input_2,decimal=6)

    # Test self.time_series()
    npt.assert_almost_equal(data.time_series(prepared=True)[0],np.array([vector,vector]).T)
    npt.assert_almost_equal(data.time_series(prepared=True)[1], np.array([vector, vector]).T)
    npt.assert_almost_equal(data.time_series(prepared=False),np.array([input_1,input_2]),decimal=6)

def test_prepare():
    """
    This function mimics the real case of HCP data, where
    we builds up two subjects with 4800 time points and 2 channels.
    These subjects are then saved in temporary text files.
    In the prepare step, we keep the first 1200 time points and 2 channels.
    standardize the data and check the output.
    Finally, we test the corresponding set_keep function.
    """
    import numpy as np
    import os

    save_dir = './test_prepare_temp/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Construct the input arrays
    vector_1 = np.array([-1.5 ** 0.5, 0, 1.5 ** 0.5])
    vector_2 = np.array([1.5 ** 0.5, 0, -1.5 ** 0.5])
    input_1_short = np.array([vector_1 + 1.0, vector_1 * 5.0]).T
    input_2_short = np.array([vector_2 * 0.1 + 1.0, vector_2 * 2.0 - 3.0]).T
    # Copy input_i_short into a (1200,2) ndarray input_i_long
    input_1_long = np.tile(input_1_short,(400,1))
    input_2_long = np.tile(input_2_short,(400,1))
    # Concatenate input_i_long with gaussian noise to get (4800,2) ndarray input_i
    gaussian_array = np.random.normal(100.,5.0,size=(3600, 2))
    input_1 = np.concatenate([input_1_long, gaussian_array], axis=0)
    input_2 = np.concatenate([input_2_long,gaussian_array],axis=0)

    # save the input_1 and input_2
    np.savetxt(f'{save_dir}10001.txt',input_1)
    np.savetxt(f'{save_dir}10002.txt',input_2)

    data = Data(save_dir)
    prepare_dict = {'select':{'timepoints':[0,1200]},
                    'standardize':{}
                    }
    data.prepare(prepare_dict)

    answer_1 = np.tile(np.array([vector_1, vector_1]).T,(400,1))
    answer_2 = np.tile(np.array([vector_2, vector_2]).T, (400, 1))
    npt.assert_almost_equal(data.arrays[0],answer_1,decimal=6)
    npt.assert_almost_equal(data.arrays[1], answer_2,decimal=6)

    with data.set_keep([0]):
        npt.assert_equal(data.keep, [0])
        for element in data.dataset(sequence_length=1200, batch_size=1):
            npt.assert_almost_equal(element['data'].numpy(), np.array([answer_1]),decimal=6)

    with data.set_keep([1]):
        npt.assert_equal(data.keep, [1])
        for element in data.dataset(sequence_length=1200, batch_size=1):
            npt.assert_almost_equal(element['data'].numpy(), np.array([answer_2]),decimal=6)


    # Remove the directory after testing
    from shutil import rmtree
    rmtree(save_dir)
