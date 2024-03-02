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

    file_path = './test_plot_violin_temp/'

    # Check if the file exists
    if os.path.exists(file_path):
        # If it exists, delete the file
        os.remove(file_path)
