import os

import numpy as np
import numpy.testing as npt

def test_rough_square_axes():
    from osl_dynamics.utils.plotting import  rough_square_axes


    # Test case 1: Even number of plots
    n_plots1 = 16
    short1, long1, empty1 = rough_square_axes(n_plots1)
    npt.assert_equal(short1, 4)
    npt.assert_equal(long1, 4)
    npt.assert_equal(empty1, 0)

    # Test case 2: Odd number of plots
    n_plots2 = 25
    short2, long2, empty2 = rough_square_axes(n_plots2)
    npt.assert_equal(short2, 5)
    npt.assert_equal(long2, 5)
    npt.assert_equal(empty2, 0)

    # Test case 3: Prime number of plots
    n_plots3 = 17
    short3, long3, empty3 = rough_square_axes(n_plots3)
    npt.assert_equal(short3, 5)
    npt.assert_equal(long3, 4)
    npt.assert_equal(empty3, 3)

    # Test case 4: Single plot
    n_plots4 = 1
    short4, long4, empty4 = rough_square_axes(n_plots4)
    npt.assert_equal(short4, 1)
    npt.assert_equal(long4, 1)
    npt.assert_equal(empty4, 0)


def test_plot_line():
    from osl_dynamics.utils.plotting import plot_line

    # Create a directory to store test plots
    test_plot_dir = 'test_plot'
    line_plot_dir = os.path.join(test_plot_dir, 'line_plot')
    if os.path.exists(line_plot_dir):
        for file in os.listdir(line_plot_dir):
            os.remove(os.path.join(line_plot_dir, file))
    else:
        os.makedirs(line_plot_dir)

    # Test cases
    # Test case 1: Basic line plot with single line
    x1 = np.linspace(0, 10, 100)
    y1 = np.sin(x1)
    plot_line([x1], [y1], filename=os.path.join(line_plot_dir, 'test_case1.png'))

    # Test case 2: Basic line plot with multiple lines and legend
    x2 = np.linspace(0, 10, 100)
    y2 = np.sin(x2)
    y3 = np.cos(x2)
    labels2 = ['Sin', 'Cos']
    plot_line([x2, x2], [y2, y3], labels=labels2, filename=os.path.join(line_plot_dir, 'test_case2.png'))

    # Test case 3: Line plot with legend, title, and axis labels
    x3 = np.linspace(0, 10, 100)
    y3 = np.sin(x3)
    plot_line([x3], [y3], labels='Sin', title='Sine Wave', x_label='X-axis', y_label='Y-axis',
              filename=os.path.join(line_plot_dir, 'test_case3.png'))

    # Test case 4: Line plot with specified ranges for x and y axes
    x4 = np.linspace(0, 10, 100)
    y4 = np.sin(x4)
    plot_line([x4], [y4], x_range=[0, 5], y_range=[-1, 1], filename=os.path.join(line_plot_dir, 'test_case4.png'))

    # Test case 5: Line plot with error bars
    x5 = np.linspace(0, 10, 100)
    y5 = np.sin(x5)
    errors_min = y5 - 0.2
    errors_max = y5 + 0.2
    plot_line([x5], [y5], errors=[[errors_min], [errors_max]], filename=os.path.join(line_plot_dir, 'test_case5.png'))


def test_plot_scatter():
    from osl_dynamics.utils.plotting import plot_scatter

    # Create scatter_plot folder if it doesn't exist
    if not os.path.exists('test_plot/scatter_plot'):
        os.makedirs('test_plot/scatter_plot')
    else:
        # Delete previous files in scatter_plot folder
        filelist = [f for f in os.listdir('test_plot/scatter_plot')]
        for f in filelist:
            os.remove(os.path.join('test_plot/scatter_plot', f))

    # Test case 1: Basic scatter plot with single data set
    x1 = np.random.rand(50)
    y1 = np.random.rand(50)
    plot_scatter([x1], [y1], filename='test_plot/scatter_plot/test_plot_scatter_1.png')

    # Test case 2: Scatter plot with multiple data sets and legend
    x2 = [np.random.rand(50) for _ in range(3)]
    y2 = [np.random.rand(50) for _ in range(3)]
    labels2 = ['Data 1', 'Data 2', 'Data 3']
    plot_scatter(x2, y2, labels=labels2, filename='test_plot/scatter_plot/test_plot_scatter_2.png')

    # Test case 3: Scatter plot with error bars
    x3 = np.random.rand(50)
    y3 = np.random.rand(50)
    errors3 = np.random.rand(50) * 0.2  # Example error bars
    plot_scatter([x3], [y3], errors=[errors3], filename='test_plot/scatter_plot/test_plot_scatter_3.png')

def test_plot_gmm():
    from osl_dynamics.utils.plotting import plot_gmm

    # Create plot_gmm folder if it doesn't exist
    if not os.path.exists('test_plot/plot_gmm'):
        os.makedirs('test_plot/plot_gmm')
    else:
        # Delete previous files in plot_gmm folder
        filelist = [f for f in os.listdir('test_plot/plot_gmm')]
        for f in filelist:
            os.remove(os.path.join('test_plot/plot_gmm', f))

    # Test case 1: Basic GMM plot
    data1 = np.random.normal(loc=0, scale=1, size=1000)
    amplitudes1 = np.array([0.5, 0.5])
    means1 = np.array([-1, 1])
    stddevs1 = np.array([1, 1])
    plot_gmm(
        data=data1,
        amplitudes=amplitudes1,
        means=means1,
        stddevs=stddevs1,
        bins=50,
        legend_loc=1,
        x_range=None,
        y_range=None,
        x_label="X",
        y_label="Density",
        title="Gaussian Mixture Model",
        filename='test_plot/plot_gmm/test_plot_gmm_1.png'
    )

    # Test case 2: GMM plot with customized style
    data2 = np.random.normal(loc=0, scale=1, size=1000)
    amplitudes2 = np.array([0.3, 0.7])
    means2 = np.array([-1, 1])
    stddevs2 = np.array([1, 1])
    fig_kwargs2 = {'figsize': (10, 6)}
    plot_gmm(
        data=data2,
        amplitudes=amplitudes2,
        means=means2,
        stddevs=stddevs2,
        bins=50,
        legend_loc=1,
        x_range=None,
        y_range=None,
        x_label="X",
        y_label="Density",
        title="Customized GMM Plot",
        fig_kwargs=fig_kwargs2,
        filename='test_plot/plot_gmm/test_plot_gmm_2.png'
    )

def test_plot_hist():
    from osl_dynamics.utils.plotting import plot_hist

    # Create hist_plot folder if it doesn't exist
    if not os.path.exists('test_plot/hist_plot'):
        os.makedirs('test_plot/hist_plot')
    else:
        # Delete previous files in hist_plot folder
        filelist = [f for f in os.listdir('test_plot/hist_plot')]
        for f in filelist:
            os.remove(os.path.join('test_plot/hist_plot', f))

    plot_kwargs = {'histtype':'bar'}

    # Test case 1: Basic histogram plot with single data set
    data1 = np.random.normal(loc=0, scale=1, size=1000)
    bins1 = 20
    plot_hist(data=[data1], bins=[bins1], plot_kwargs=plot_kwargs, filename='test_plot/hist_plot/test_plot_hist_1.png')

    # Test case 2: Histogram plot with multiple data sets and legend
    data2 = [np.random.normal(loc=i, scale=1, size=100) for i in range(3)]
    bins2 = [20, 15, 10]
    labels2 = ['Data 1', 'Data 2', 'Data 3']
    plot_hist(data=data2, bins=bins2, labels=labels2, plot_kwargs=plot_kwargs, filename='test_plot/hist_plot/test_plot_hist_2.png')

    # Test case 3: Histogram plot with customized plot style
    data3 = [np.random.normal(loc=i, scale=1, size=1000) for i in range(2)]
    bins3 = [20, 15]
    plot_kwargs3 = {'histtype': 'bar', 'alpha': 0.5,}
    plot_hist(data=data3, bins=bins3, plot_kwargs=plot_kwargs3, filename='test_plot/hist_plot/test_plot_hist_3.png')


def test_plot_violin():
    from osl_dynamics.utils.plotting import plot_violin
    # Create plot_violin folder if it doesn't exist
    if not os.path.exists('test_plot/plot_violin'):
        os.makedirs('test_plot/plot_violin')
    else:
        # Delete previous files in plot_violin folder
        filelist = [f for f in os.listdir('test_plot/plot_violin')]
        for f in filelist:
            os.remove(os.path.join('test_plot/plot_violin', f))

    # Generate sample 2D data
    np.random.seed(0)
    n_sessions = 100
    n_states = 2
    data = np.zeros((n_sessions, n_states))

    # Populate data[:,0] with a normal distribution with mean 0.1 and std 0.1
    data[:, 0] = np.random.normal(loc=0.1, scale=0.1, size=n_sessions)

    # Populate data[:,1] with a normal distribution with mean 0.0 and std 0.1
    data[:, 1] = np.random.normal(loc=0.0, scale=0.1, size=n_sessions)


    # Test the function
    plot_violin(data.T, x_label="State", y_label="Value", title="Violin Plot",filename='test_plot/plot_violin/toy_example.png')

def test_plot_time_series():
    from osl_dynamics.utils.plotting import plot_time_series

    # Create directory if it doesn't exist
    save_dir = 'test_plot/plot_time_series/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        # Delete previous files in the directory
        filelist = [f for f in os.listdir(save_dir)]
        for f in filelist:
            os.remove(os.path.join(save_dir, f))

    # Generate synthetic time series data
    n_samples = 1000
    n_channels = 3
    time_series = np.random.randn(n_samples, n_channels)

    # Test case 1: Basic plot with default parameters
    plot_time_series(time_series, filename=os.path.join(save_dir, 'plot_basic.png'))

    # Test case 2: Plot with specified number of samples and y_tick_values
    n_samples_plot = 500
    y_tick_values = ['Channel A', 'Channel B', 'Channel C']
    plot_time_series(time_series, n_samples=n_samples_plot, y_tick_values=y_tick_values,
                              filename=os.path.join(save_dir, 'plot_samples_ticks.png'))

    # Test case 3: Plot with customized plot appearance
    plot_kwargs = {'lw': 1.2, 'color': 'red'}
    plot_time_series(time_series, plot_kwargs=plot_kwargs, filename=os.path.join(save_dir, 'plot_custom.png'))

def test_plot_separate_time_series():
    from osl_dynamics.utils.plotting import plot_separate_time_series

    # Create directory if it doesn't exist
    save_dir = 'test_plot/plot_separate_time_series/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        # Delete previous files in the directory
        filelist = [f for f in os.listdir(save_dir)]
        for f in filelist:
            os.remove(os.path.join(save_dir, f))

    # Generate synthetic time series data
    n_samples = 1000
    n_lines = 3
    time_series1 = np.random.randn(n_samples, n_lines)
    time_series2 = np.random.randn(n_samples, n_lines)

    # Test case 1: Basic plot with default parameters
    plot_separate_time_series(time_series1, filename=os.path.join(save_dir, 'plot_basic.png'))

    # Test case 2: Plot with specified number of samples and sampling frequency
    sampling_frequency = 100  # Hz
    plot_separate_time_series(time_series1, n_samples=500, sampling_frequency=sampling_frequency,
                                    filename=os.path.join(save_dir, 'plot_samples_frequency.png'))

    # Test case 3: Plot with customized appearance
    plot_kwargs = {'lw': 1.2, 'color': 'red'}
    plot_separate_time_series(time_series2, plot_kwargs=plot_kwargs,
                                    filename=os.path.join(save_dir, 'plot_custom.png'))

def test_plot_epoched_time_series():
    from osl_dynamics.utils.plotting import plot_epoched_time_series

    # Create directory if it doesn't exist
    save_dir = 'test_plot/plot_epoched_time_series/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        # Delete previous files in the directory
        filelist = [f for f in os.listdir(save_dir)]
        for f in filelist:
            os.remove(os.path.join(save_dir, f))

    # Generate synthetic continuous data
    n_samples = 10000
    n_channels = 4
    continuous_data = np.random.normal(loc=1.0, scale=1.0, size=(n_samples, n_channels))

    # Generate synthetic time index for epochs
    epoch_length = 200
    time_index = np.arange(epoch_length, n_samples, epoch_length)

    # Test case 1: Basic plot with default parameters
    plot_epoched_time_series(continuous_data, time_index, filename=os.path.join(save_dir, 'plot_basic.png'))

    # Test case 2: Plot with baseline correction
    plot_epoched_time_series(continuous_data, time_index, baseline_correct=True,
                             filename=os.path.join(save_dir, 'plot_baseline_corrected.png'))

    # Test case 3: Plot with legend and custom appearance
    plot_kwargs = {'lw': 1.2, 'color': 'red'}
    plot_epoched_time_series(continuous_data, time_index, legend=True, plot_kwargs=plot_kwargs,
                             filename=os.path.join(save_dir, 'plot_legend_custom.png'))

def test_plot_matrices():
    from osl_dynamics.utils.plotting import plot_matrices

    # Create directory and delete existing files if it already exists
    plot_dir = 'test_plot/plot_matrices'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir, exist_ok=True)
    else:
        for file in os.listdir(plot_dir):
            os.remove(os.path.join(plot_dir, file))

    # Test case 1: Single matrix plot
    matrix1 = np.random.rand(15, 15)
    plot_matrices([matrix1], main_title="Single Matrix Plot", titles=["Matrix 1"],
                  filename=os.path.join(plot_dir, 'test_single_matrix.png'))

    # Test case 2: Multiple matrices plot
    matrix2 = np.random.rand(15, 15)
    matrix3 = np.random.rand(15, 15)
    matrices = [matrix1, matrix2, matrix3]
    plot_matrices(matrices, main_title="Collection of Matrices",
                  titles=["Matrix 1", "Matrix 2", "Matrix 3"],
                  filename=os.path.join(plot_dir, 'test_multiple_matrices.png'))

    # Test case 3: Matrices with log normalization
    matrix4 = np.random.rand(10, 10)
    matrix5 = np.random.rand(10, 10)
    matrix6 = np.random.rand(10, 10)
    matrices_log = [matrix4, matrix5, matrix6]
    plot_matrices(matrices_log, main_title="Matrices with Log Normalization",
                  titles=["Matrix 4", "Matrix 5", "Matrix 6"],
                  log_norm=True,
                  filename=os.path.join(plot_dir, 'test_matrices_log_normalization.png'))

def test_plot_connections():
    from osl_dynamics.utils.plotting import plot_connections

    def generate_weights_matrix(shape):
        # Generate a random matrix with the specified shape
        return np.random.rand(*shape)

    plot_dir = 'test_plot/plot_connections'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    else:
        for file in os.listdir(plot_dir):
            os.remove(os.path.join(plot_dir, file))

    # Test case 1: Square weights matrix
    shape_1 = (3, 3)
    weights_1 = np.array([[1.0,2.0,0.0],
                          [2.0,1.0,-0.5],
                          [0.0,-0.5,1.0]])

    plot_connections(weights_1, labels=[f"Node {i}" for i in range(1, shape_1[0] + 1)],
                     filename=os.path.join(plot_dir, f'test_connections_{shape_1[0]}x{shape_1[1]}.png'))

    # Test case 2: Rectangular weights matrix
    shape_2 = (8,8)
    weights_2 = generate_weights_matrix(shape_2)
    plot_connections(weights_2, labels=[f"Node {i}" for i in range(1, shape_2[0] + 1)],
                     filename=os.path.join(plot_dir, f'test_connections_{shape_2[0]}x{shape_2[1]}.png'))

    # Test case 3: Large square weights matrix
    shape_3 = (10, 10)
    weights_3 = generate_weights_matrix(shape_3)
    plot_connections(weights_3, labels=[f"Node {i}" for i in range(1, shape_3[0] + 1)],
                     filename=os.path.join(plot_dir, f'test_connections_{shape_3[0]}x{shape_3[1]}.png'))

def test_plot_alpha():
    from osl_dynamics.utils.plotting import plot_alpha

    def generate_alpha(n_samples=100, n_modes=3, n_alphas=2):
        """Generate example alpha matrices."""
        alphas = []
        for _ in range(n_alphas):
            alpha = np.random.randn(n_samples, n_modes)
            alphas.append(alpha)
        return alphas

    plot_dir = 'test_plot/plot_alpha'
    if os.path.exists(plot_dir):
        for file in os.listdir(plot_dir):
            os.remove(os.path.join(plot_dir, file))
    else:
        os.makedirs(plot_dir)

    alphas = generate_alpha()

    # Ensure the plot directory exists or create it
    os.makedirs(plot_dir, exist_ok=True)

    # Test case 1: Basic plot
    plot_alpha(*alphas, filename=os.path.join(plot_dir, 'basic_plot.png'))

    # Test case 2: Plot with customized parameters
    plot_alpha(
        *alphas,
        n_samples=150,
        #cmap="Blues",
        sampling_frequency=1000,
        y_labels=["Alpha 1", "Alpha 2"],
        title="Example Alpha Plot",
        filename=os.path.join(plot_dir, 'custom_alpha_plot.png')
    )

def test_plot_state_lifetimes():
    from osl_dynamics.utils.plotting import plot_state_lifetimes

    # Create directory for test plots if it doesn't exist
    plot_dir = "test_plot/plot_state_lifetimes"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    else:
        # Delete existing files in the directory
        for file_name in os.listdir(plot_dir):
            file_path = os.path.join(plot_dir, file_name)
            os.remove(file_path)

    # Test case 1: Basic input
    state_time_course = np.random.randint(0, 2, size=(100, 5))  # Generate random state time course
    filename = os.path.join(plot_dir, "test_basic.png")  # Define output filename
    plot_state_lifetimes(state_time_course, filename=filename)  # Plot and save

    # Test case 2: Customized parameters
    filename = os.path.join(plot_dir, "test_customized.png")  # Define output filename
    x_label = "State Lifetime"  # Customize x-axis label
    y_label = "Frequency"  # Customize y-axis label
    plot_kwargs = {"alpha": 0.5}  # Customize plot appearance
    fig_kwargs = {"figsize": (8, 6)}  # Customize figure size
    plot_state_lifetimes(state_time_course, x_label=x_label, y_label=y_label,
                         plot_kwargs=plot_kwargs, fig_kwargs=fig_kwargs,
                         filename=filename)  # Plot and save
