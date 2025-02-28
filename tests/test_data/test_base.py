import os
import pytest
import numpy as np
import numpy.testing as npt
from osl_dynamics.data import Data


def test_data_nparray():
    # Note: the input should be (n_timepoints,n_channels)
    input = np.array([[1., 0.5], [2., 0.4], [1.5, 0.3]])

    data = Data(input, buffer_size=1000, sampling_frequency=2.0)

    # Test self.__iter__
    for obj in data:
        npt.assert_almost_equal(obj, input)
    # Test self.__getitem__
    npt.assert_almost_equal(data.arrays[0], input)
    # Test @property self.raw_data
    npt.assert_almost_equal(data.raw_data[0], input)
    # Test @property self.n_channels
    npt.assert_equal(data.n_channels, 2)
    # Test @property self.n_samples
    npt.assert_equal(data.n_samples, 3)
    # Test set.sampling_frequency
    npt.assert_equal(data.sampling_frequency, 2.0)
    data.set_sampling_frequency(1.0)
    npt.assert_equal(data.sampling_frequency, 1.0)
    # Test set buffer_size
    npt.assert_equal(data.buffer_size, 1000)
    data.set_buffer_size(100)
    npt.assert_equal(data.buffer_size, 100)

    data.save()
    array_save = np.load('./array0.npy')
    npt.assert_almost_equal(array_save,input)
    os.remove('./array0.npy')
    os.remove('preparation.pkl')
    # Test self.select
    data.select(channels=[1], timepoints=[1, 4], sessions=[0])
    npt.assert_almost_equal(data.arrays[0], input[1:3, 1:])
    data.delete_dir()


def test_data_list_nparray():
    input_1 = np.array([[1., 0.5], [2., 0.4], [1.5, 0.3]])
    input_2 = input_1 / 2
    data = Data([input_1, input_2])
    # Test self.__getitem__
    npt.assert_almost_equal(data.arrays[0], input_1)
    npt.assert_almost_equal(data.arrays[1], input_2)
    # Test @property self.n_sessions
    npt.assert_equal(data.n_sessions, 2)
    # Test @property self.n_samples
    npt.assert_equal(data.n_samples, 6)
    # Test @self.time_series()
    npt.assert_almost_equal(data.time_series(concatenate=True),np.concatenate([input_1,input_2]))
    npt.assert_almost_equal(data.time_series(concatenate=False), np.array([input_1, input_2]))

    # Test data.set_keep
    with data.set_keep([0]):
        npt.assert_equal(data.keep, [0])
        for element in data.dataset(sequence_length=3, batch_size=1):
            npt.assert_almost_equal(element['data'].numpy(), np.array([input_1]))

    with data.set_keep([1]):
        npt.assert_equal(data.keep, [1])
        for element in data.dataset(sequence_length=3, batch_size=1):
            npt.assert_almost_equal(element['data'].numpy(), np.array([input_2]))

    with data.set_keep([1, 0]):
        npt.assert_equal(data.keep, [1, 0])
        for element in data.dataset(sequence_length=3, batch_size=2, shuffle=False):
            npt.assert_almost_equal(element['data'].numpy(), np.array([input_1, input_2]))

    # Test self.select
    data.select(channels=[1], sessions=[1], timepoints=[1, 3])
    npt.assert_almost_equal(data.arrays[0], input_2[1:3, 1:])
    # Test @property self.n_samples after selection
    npt.assert_equal(data.n_samples, 2)
    data.delete_dir()

    input_3 = np.array([[1.], [2.], [1.5]])

    with pytest.raises(ValueError, match="All inputs should have the same number of channels."):
        data_wrong = Data([input_1, input_3])  # Should raise ValueError
        data_wrong.delete_dir()

def test_data_files():
    import os
    import numpy as np
    import shutil
    save_dir = './test_temp/'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    input_1 = np.array([[1., 0.5], [2., 0.4], [1.5, 0.3]])
    input_2 = input_1 / 2
    np.save(f'{save_dir}/10001.npy', input_1)
    np.savetxt(f'{save_dir}/10002.txt', input_2)

    # Case 1: A directory
    data_1 = Data(save_dir)
    npt.assert_almost_equal(data_1.arrays, [input_1, input_2])
    data_1.delete_dir()

    # Case 2: A list of file names
    file_path = [f'{save_dir}/10001.npy', f'{save_dir}/10002.txt']
    data_2 = Data(file_path)
    npt.assert_almost_equal(data_2.arrays, [input_1, input_2])
    data_2.delete_dir()

    # Case 3: Save the list of file names to a txt file
    output_txt = "./file_list.txt"  # Define output file

    with open(output_txt, "w") as f:
        for path in file_path:
            f.write(path + "\n")  # Write each item followed by a newline
    data_3 = Data(output_txt)
    npt.assert_almost_equal(data_3.arrays, [input_1, input_2])
    data_3.delete_dir()
    os.remove(output_txt)

    # Remove the directory after testing
    from shutil import rmtree
    rmtree(save_dir)


def test_standardize():
    import numpy as np
    vector = np.array([-1.5 ** 0.5, 0, 1.5 ** 0.5])
    input_1 = np.array([vector, vector + 10.0]).T
    input_2 = np.array([vector * 0.5 + 1., vector * 100]).T

    ### Case 1: input_1 & input_2 represent single session respectively
    data = Data([input_1, input_2])
    data.prepare({'standardize': {}})

    npt.assert_almost_equal(data.arrays[0], np.array([vector, vector]).T)
    npt.assert_almost_equal(data.arrays[1], np.array([vector, vector]).T)
    npt.assert_almost_equal(data.raw_data[0], input_1, decimal=6)
    npt.assert_almost_equal(data.raw_data[1], input_2, decimal=6)

    # Test self.time_series()
    npt.assert_almost_equal(data.time_series(prepared=True)[0], np.array([vector, vector]).T)
    npt.assert_almost_equal(data.time_series(prepared=True)[1], np.array([vector, vector]).T)
    npt.assert_almost_equal(data.time_series(prepared=False), np.array([input_1, input_2]), decimal=6)
    data.delete_dir()

    ### Case 2: input represents multiple sessions
    input = np.concatenate((input_1, input_2), axis=0)
    data = Data([input, input])
    data.prepare({'standardize': {'session_length': 3}})

    npt.assert_almost_equal(data.arrays[0][:3], np.array([vector, vector]).T)
    npt.assert_almost_equal(data.arrays[0][3:], np.array([vector, vector]).T)
    npt.assert_almost_equal(data.raw_data[0][:3], input_1, decimal=6)
    npt.assert_almost_equal(data.raw_data[0][3:], input_2, decimal=6)
    npt.assert_almost_equal(data.arrays[1][:3], np.array([vector, vector]).T)
    npt.assert_almost_equal(data.arrays[1][3:], np.array([vector, vector]).T)
    npt.assert_almost_equal(data.raw_data[1][:3], input_1, decimal=6)
    npt.assert_almost_equal(data.raw_data[1][3:], input_2, decimal=6)

    # Test self.time_series()
    npt.assert_almost_equal(data.time_series(prepared=True)[0][:3], np.array([vector, vector]).T)
    npt.assert_almost_equal(data.time_series(prepared=True)[0][3:], np.array([vector, vector]).T)
    npt.assert_almost_equal(data.time_series(prepared=False)[0], np.concatenate((input_1, input_2), axis=0), decimal=6)
    npt.assert_almost_equal(data.time_series(prepared=True)[1][:3], np.array([vector, vector]).T)
    npt.assert_almost_equal(data.time_series(prepared=True)[1][3:], np.array([vector, vector]).T)
    npt.assert_almost_equal(data.time_series(prepared=False)[1], np.concatenate((input_1, input_2), axis=0), decimal=6)
    data.delete_dir()

def test_filter():
    """
    This function aims to test the band-pass filter of the Data Class.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    save_dir = './test_filter_temp/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Parameters
    length = 2400
    sampling_frequency = 1 / 0.7
    cutoff_frequency = 0.15

    def generate_time_series(length, sampling_frequency):
        t = np.arange(0, length * sampling_frequency, sampling_frequency)
        # Create a signal with low and high frequency components
        low_freq_signal = np.sin(2 * np.pi * 0.05 * t)  # Low frequency component (0.1 Hz)
        high_freq_signal = np.sin(2 * np.pi * 0.3 * t)  # High frequency component (0.3 Hz)
        return low_freq_signal + high_freq_signal

    time_series = generate_time_series(length, sampling_frequency)
    time_series_save = np.column_stack((time_series, time_series))

    # save the input_1 and input_2
    np.savetxt(f'{save_dir}10001.txt', time_series_save)
    np.savetxt(f'{save_dir}10002.txt', time_series_save)

    data = Data(save_dir, sampling_frequency=sampling_frequency)
    prepare_dict = {'select': {'timepoints': [0, 2400]},
                    'filter': {'low_freq': cutoff_frequency},
                    }
    data.prepare(prepare_dict)

    filtered_ts = np.squeeze(data.time_series()[0][:, 0])

    def get_fft_magnitudes(signal, sampling_frequency):
        fft_values = np.fft.rfft(signal)
        fft_frequencies = np.fft.rfftfreq(len(signal), d=sampling_frequency)
        return fft_frequencies, np.abs(fft_values)

    def plot_frequency(fft_frequencies,fft_absolute_values):
        plt.figure(figsize=(5, 4))
        plt.plot(fft_frequencies, fft_absolute_values)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.show()


    # Compute FFT for original and filtered signals
    fft_freqs, fft_orig = get_fft_magnitudes(time_series, sampling_frequency)
    _, fft_filtered = get_fft_magnitudes(filtered_ts, sampling_frequency)

    plot_frequency(fft_freqs, fft_orig)
    plot_frequency(fft_freqs, fft_filtered)

    # Find indices corresponding to 0.05Hz and 0.3Hz
    idx_005Hz = np.argmin(np.abs(fft_freqs - 0.05))
    idx_03Hz = np.argmin(np.abs(fft_freqs - 0.3))

    # **Test: Ensure the 0.05Hz component is removed**
    npt.assert_array_less(fft_filtered[idx_005Hz], fft_orig[idx_005Hz] * 0.1,
                          err_msg="Low-frequency component (0.05Hz) was not sufficiently filtered.")

    # **Test: Ensure the 0.3Hz component is preserved within 1% relative error**
    relative_error = np.abs(fft_filtered[idx_03Hz] - fft_orig[idx_03Hz]) / fft_orig[idx_03Hz]
    npt.assert_array_less(relative_error, 0.01,
                      err_msg="High-frequency component (0.3Hz) relative error exceeds 1%.")
    data.delete_dir()

    # Remove the directory after testing
    from shutil import rmtree
    rmtree(save_dir)
def test_filter_gaussian_weighted_least_squares_straight_line_fitting_session():
    """
    This function aims to test the gaussian_weighted_least_squares_straight_line_fitting.
    It also tests the effect of "session length" variable.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    save_dir = './test_filter_temp_1/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sampling_frequency = 0.1

    time_series = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    time_series_save = np.column_stack((time_series, time_series))

    # Save the combined time series for testing
    np.savetxt(f'{save_dir}10001.txt', time_series_save)
    np.savetxt(f'{save_dir}10002.txt', time_series_save)

    data = Data(save_dir, sampling_frequency=sampling_frequency)

    # Prepare with session input
    prepare_with_session = {'filter': {'sigma': 40, 'session_length': 5}}
    data.prepare(prepare_with_session)
    filtered_ts_with_session = np.squeeze(data.time_series()[0][:, 0])

    answer = time_series - np.array([2.91948343, 2.95023502, 3., 3.04976498, 3.08051657,
                                     2.91948343, 2.95023502, 3., 3.04976498, 3.08051657])

    npt.assert_almost_equal(filtered_ts_with_session, answer)

    # Clean up temporary files
    for file in os.listdir(save_dir):
        os.remove(os.path.join(save_dir, file))
    os.rmdir(save_dir)

    # Visual illustration
    def generate_fmri_like_signal(length=400, sampling_frequency=1 / 0.72):
        time = np.arange(length) / sampling_frequency  # Time vector
        freq = np.fft.rfftfreq(length, d=1 / sampling_frequency)  # Frequency vector

        # Generate random phases
        phases = np.random.uniform(0, 2 * np.pi, len(freq))

        # Create 1/f amplitude spectrum
        amplitude = 1 / np.maximum(freq, 1)  # Avoid division by zero

        # Construct Fourier coefficients
        real_part = amplitude * np.cos(phases)
        imag_part = amplitude * np.sin(phases)

        # Create complex spectrum
        spectrum = real_part + 1j * imag_part

        # Perform inverse FFT to get time-domain signal
        signal = np.fft.irfft(spectrum, n=length)

        # Normalize to have zero mean and unit variance
        signal = (signal - np.mean(signal)) / np.std(signal)

        return time, signal

    save_dir = './test_filter_temp_2/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sampling_frequency = 1 / 0.72

    # Generate the fMRI-like signal
    time, signal = generate_fmri_like_signal(sampling_frequency=sampling_frequency)

    time_series_save = np.column_stack((signal, signal))

    # Save the combined time series for testing
    np.savetxt(f'{save_dir}10001.txt', time_series_save)
    np.savetxt(f'{save_dir}10002.txt', time_series_save)

    data = Data(save_dir, sampling_frequency=sampling_frequency)

    # Prepare with session input
    prepare_with_session = {'filter': {'sigma': 50, 'session_length': 400}}
    data.prepare(prepare_with_session)

    filtered_ts_with_session = np.squeeze(data.time_series()[0][:, 0])

    # Plot the signal
    plt.figure(figsize=(10, 4))
    plt.plot(time, signal, label='Simulated fMRI Signal', linewidth=1)
    plt.plot(time, filtered_ts_with_session, label='After filtering', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Synthetic fMRI-like Time Series')
    plt.legend()
    plt.show()
    data.delete_dir()
    # Remove the directory after testing
    from shutil import rmtree
    rmtree(save_dir)

def test_prepare_method_order():
    """
    This function tests how method order would impact the 'prepare' method
    """
    import numpy as np
    import os
    vector_1 = np.array([1.0,2.0,3.0,4.0,5.0,-10.0])
    vector_2 = np.array([0.1,1000.,-10000.,30.,-25.3,0.0])
    input = np.array([vector_1,vector_2]).T

    ### Case 1
    data_1 = Data([input,input],sampling_frequency=0.1)
    prepare_kwargs_1 = {'method_order':['select','filter','standardize'],
                        'select':{'channels':[0],'sessions':[1],'timepoints':[0,5]},
                        'filter':{'sigma': 40},
                        'standardize':{}}
    data_1.prepare(prepare_kwargs_1)
    npt.assert_equal(data_1.n_sessions,1)
    answer_1 = np.array([[-1.41700965],
                         [-0.70148675],
                         [0.],
                         [0.70148675],
                         [1.41700965]])
    npt.assert_almost_equal(data_1.arrays[0],answer_1)
    data_1.delete_dir()

    ### Case 2
    data_2 = Data([input, input], sampling_frequency=0.1)
    prepare_kwargs_2 = {'method_order': ['select','standardize','filter'],
                        'select': {'channels': [0], 'sessions': [1], 'timepoints': [0, 5]},
                        'filter': {'sigma': 40},
                        'standardize': {}}
    data_2.prepare(prepare_kwargs_2)
    answer_2 = np.array([[-1.35727975],
                         [-0.67191763],
                         [0.],
                         [0.67191763],
                         [1.35727975]])
    npt.assert_almost_equal(data_2.arrays[0], answer_2)
    data_2.delete_dir()

    ### Case 3
    data_3 = Data([input, input], sampling_frequency=0.1)
    prepare_kwargs_3 = {'method_order': ['standardize', 'filter','select'],
                        'select': {'channels': [0], 'sessions': [1], 'timepoints': [0, 5]},
                        'filter': {'sigma': 40},
                        'standardize': {}}
    data_3.prepare(prepare_kwargs_3)
    answer_3 = np.array([[-0.02573626],
                         [0.18943394],
                         [0.41620497],
                         [0.64729963],
                         [0.87427632]])
    npt.assert_almost_equal(data_3.arrays[0], answer_3)
    data_3.delete_dir()

    ### Case 4
    data_4 = Data([input, input], sampling_frequency=0.1)
    prepare_kwargs_4 = {'select_1': {'channels': [0], 'sessions': [1], 'timepoints': [0, 5]},
                        'filter_123': {'sigma': 40},
                        'standardize_432': {}}
    data_4.prepare(prepare_kwargs_4)
    npt.assert_equal(data_4.n_sessions, 1)
    answer_4 = np.array([[-1.41700965],
                         [-0.70148675],
                         [0.],
                         [0.70148675],
                         [1.41700965]])
    npt.assert_almost_equal(data_4.arrays[0], answer_4)
    data_4.delete_dir()

    ### Case 5
    data_5 = Data([input, input], sampling_frequency=0.1)
    prepare_kwargs_5 = {'select': {'channels': [0], 'sessions': [1], 'timepoints': [0, 5]},
                        'standardize': {},
                        'filter': {'sigma': 40}}
    data_5.prepare(prepare_kwargs_5)
    answer_5 = np.array([[-1.35727975],
                         [-0.67191763],
                         [0.],
                         [0.67191763],
                         [1.35727975]])
    npt.assert_almost_equal(data_5.arrays[0], answer_5)
    data_5.delete_dir()

    ### Case 6
    data_6 = Data([input, input], sampling_frequency=0.1)
    prepare_kwargs_6 = {'standardize': {},
                        'filter': {'sigma': 40},
                        'select': {'channels': [0], 'sessions': [1], 'timepoints': [0, 5]},
                        }
    data_6.prepare(prepare_kwargs_6)
    answer_6 = np.array([[-0.02573626],
                         [0.18943394],
                         [0.41620497],
                         [0.64729963],
                         [0.87427632]])
    npt.assert_almost_equal(data_6.arrays[0], answer_6)
    data_6.delete_dir()


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
    input_1_long = np.tile(input_1_short, (400, 1))
    input_2_long = np.tile(input_2_short, (400, 1))
    # Concatenate input_i_long with gaussian noise to get (4800,2) ndarray input_i
    gaussian_array = np.random.normal(100., 5.0, size=(3600, 2))
    input_1 = np.concatenate([input_1_long, gaussian_array], axis=0)
    input_2 = np.concatenate([input_2_long, gaussian_array], axis=0)

    # save the input_1 and input_2
    np.savetxt(f'{save_dir}10001.txt', input_1)
    np.savetxt(f'{save_dir}10002.txt', input_2)

    data = Data(save_dir)
    prepare_dict = {'select': {'timepoints': [0, 1200]},
                    'standardize': {}
                    }
    data.prepare(prepare_dict)

    answer_1 = np.tile(np.array([vector_1, vector_1]).T, (400, 1))
    answer_2 = np.tile(np.array([vector_2, vector_2]).T, (400, 1))
    npt.assert_almost_equal(data.arrays[0], answer_1, decimal=6)
    npt.assert_almost_equal(data.arrays[1], answer_2, decimal=6)

    with data.set_keep([0]):
        npt.assert_equal(data.keep, [0])
        for element in data.dataset(sequence_length=1200, batch_size=1):
            npt.assert_almost_equal(element['data'].numpy(), np.array([answer_1]), decimal=6)

    with data.set_keep([1]):
        npt.assert_equal(data.keep, [1])
        for element in data.dataset(sequence_length=1200, batch_size=1):
            npt.assert_almost_equal(element['data'].numpy(), np.array([answer_2]), decimal=6)

    # Remove the directory after testing
    from shutil import rmtree
    rmtree(save_dir)



