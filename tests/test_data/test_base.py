import numpy as np
import numpy.testing as npt
from osl_dynamics.data import Data


def test_data_nparray():
    # Note: the input should be (n_timepoints,n_channels)
    input = np.array([[1., 0.5], [2., 0.4], [1.5, 0.3]])

    data = Data(input,buffer_size=1000,sampling_frequency=2.0)

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
    # Test self.select
    data.select(channels=[1], timepoints=[1, 4],sessions=[0])
    npt.assert_almost_equal(data.arrays[0], input[1:3, 1:])


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
        for element in data.dataset(sequence_length=3, batch_size=2):
            print(element['data'].numpy().shape)
            print(element['data'].numpy())

            npt.assert_almost_equal(element['data'].numpy(), np.array([input_2, input_1]))

    # Test self.select
    data.select(channels=[1],sessions=[1],timepoints=[1,3])
    npt.assert_almost_equal(data.arrays[0], input_2[1:3, 1:])
    # Test @property self.n_samples after selection
    npt.assert_equal(data.n_samples, 2)


def test_data_files():
    import os
    import numpy as np
    save_dir = './test_temp/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    input_1 = np.array([[1., 0.5], [2., 0.4], [1.5, 0.3]])
    input_2 = input_1 / 2
    np.save(f'{save_dir}10001.npy', input_1)
    np.savetxt(f'{save_dir}10002.txt', input_2)
    data = Data(save_dir)
    npt.assert_almost_equal(data.arrays, [input_1, input_2])

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
    cutoff_frequency = 0.25

    def generate_time_series(length, sampling_frequency):
        t = np.arange(0, length * sampling_frequency, sampling_frequency)
        # Create a signal with low and high frequency components
        low_freq_signal = np.sin(2 * np.pi * 0.1 * t)  # Low frequency component (0.1 Hz)
        high_freq_signal = np.sin(2 * np.pi * 0.3 * t)  # High frequency component (1.0 Hz)
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

    def plot_frequency(signal, sampling_frequency):
        fft_values = np.fft.rfft(signal)
        fft_frequencies = np.fft.rfftfreq(len(signal), d=sampling_frequency)

        plt.figure(figsize=(5, 4))
        plt.plot(fft_frequencies, np.abs(fft_values))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.show()

    plot_frequency(time_series, sampling_frequency)
    plot_frequency(filtered_ts, sampling_frequency)

    # npt.assert_array_less(np.abs(np.fft.rfft(filtered_ts)), 0.2)


def test_filter_session():
    """
    This function aims to test the band-pass filter of the Data Class when session input is fed in
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    save_dir = './test_filter_temp/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Parameters
    length = 1200
    sampling_frequency = 1 / 0.7
    cutoff_frequency = 0.2

    def generate_time_series(length, sampling_frequency):
        t = np.arange(0, length * sampling_frequency, sampling_frequency)
        # Create distinct signals for two sessions
        session_1_signal = np.sin(2 * np.pi * 0.1 * t)  # Low frequency component (0.1 Hz)
        session_2_signal = np.sin(2 * np.pi * 0.4 * t)  # High frequency component (0.4 Hz)

        # Combine into one time series with two sessions
        combined_signal = np.hstack((session_1_signal, session_2_signal))
        return combined_signal

    time_series = generate_time_series(length, sampling_frequency)
    time_series_save = np.column_stack((time_series, time_series))

    # Save the combined time series for testing
    np.savetxt(f'{save_dir}10001.txt', time_series_save)
    np.savetxt(f'{save_dir}10002.txt', time_series_save)

    # Initialize the Data object
    data = Data(save_dir, sampling_frequency=sampling_frequency)

    # Prepare without session input
    prepare_no_session = {'select': {'timepoints': [0, 2400]},
                          'filter': {'low_freq': cutoff_frequency}}
    data.prepare(prepare_no_session)
    filtered_ts_no_session = np.squeeze(data.time_series()[0][:,0])

    # Prepare with session input
    prepare_with_session = {'select': {'timepoints': [0, 2400]},
                            'filter': {'low_freq': cutoff_frequency, 'session_length': 1200}}
    data.prepare(prepare_with_session)
    filtered_ts_with_session = np.squeeze(data.time_series()[0][:,0])

    # Plot frequency spectrum
    def plot_frequency(signal, sampling_frequency, title):
        fft_values = np.fft.rfft(signal)
        fft_frequencies = np.fft.rfftfreq(len(signal), d=sampling_frequency)

        plt.figure(figsize=(5, 4))
        plt.plot(fft_frequencies, np.abs(fft_values))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title(title)
        plt.show()

    plt.plot(filtered_ts_no_session)
    plt.title('Time domain, filtered signal without Session')
    plt.show()
    plt.plot(filtered_ts_with_session)
    plt.title('Time domain, filtered signal with session')
    plt.show()

    # Plot original, filtered without session, and filtered with session
    #plot_frequency(time_series_combined, sampling_frequency, "Original Signal")
    plot_frequency(filtered_ts_no_session, sampling_frequency, "Filtered Signal Without Session")
    plot_frequency(filtered_ts_with_session, sampling_frequency, "Filtered Signal With Session")

    # Clean up temporary files
    for file in os.listdir(save_dir):
        os.remove(os.path.join(save_dir, file))
    os.rmdir(save_dir)

def test_filter_gaussian_weighted_least_squares_straight_line_fitting():
    """
    This function aims to test the gaussian_weighted_least_squares_straight_line_fitting
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    save_dir = './test_filter_temp_1/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sampling_frequency = 0.1

    time_series = np.array([1.0,2.0,3.0,4.0,5.0,1.0,2.0,3.0,4.0,5.0])
    time_series_save = np.column_stack((time_series, time_series))

    # Save the combined time series for testing
    np.savetxt(f'{save_dir}10001.txt', time_series_save)
    np.savetxt(f'{save_dir}10002.txt', time_series_save)

    data = Data(save_dir, sampling_frequency=sampling_frequency)

    # Prepare with session input
    prepare_with_session = {'filter': {'sigma': 40, 'session_length':5}}
    data.prepare(prepare_with_session)
    filtered_ts_with_session = np.squeeze(data.time_series()[0][:, 0])

    answer = time_series - np.array([2.91948343,  2.95023502,  3.        ,  3.04976498,  3.08051657,
                                     2.91948343,  2.95023502,  3.        ,  3.04976498,  3.08051657])

    npt.assert_almost_equal(filtered_ts_with_session,answer)


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

    sampling_frequency = 1/0.72

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
    plt.plot(time,filtered_ts_with_session,label='After filtering',linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Synthetic fMRI-like Time Series')
    plt.legend()
    plt.show()