import numpy as np
import numpy.testing as npt


def test_stdcor2cov():
    from rotation.utils import stdcor2cov
    stds = np.array([[4.0,2.0],[10.0,20.0]])
    corrs = np.array([[[1.0,0.5],[0.5,1.0]],[[1.0,-0.2],[-0.2,1.0]]])
    covs = stdcor2cov(stds, corrs)
    npt.assert_equal(covs,np.array([[[16.0,4.0],[4.0,4.0]],[[100.0,-40.0],[-40.0,400.0]]]))

    stds = np.array([[[4.0,0.0],[0.0,2.0]],[[10.0,0.0],[0.0,20.0]]])
    covs = stdcor2cov(stds,corrs)
    npt.assert_equal(covs, np.array([[[16.0, 4.0], [4.0, 4.0]], [[100.0, -40.0], [-40.0, 400.0]]]))

def test_cov2stdcor():
    from rotation.utils import cov2stdcor
    covs = np.array([[[16.0, 4.0], [4.0, 4.0]], [[100.0, -40.0], [-40.0, 400.0]]])
    stds, corrs = cov2stdcor(covs)
    npt.assert_equal(stds, np.array([[4.0,2.0],[10.0,20.0]]))
    npt.assert_equal(corrs, np.array([[[1.0,0.5],[0.5,1.0]],[[1.0,-0.2],[-0.2,1.0]]]))
def test_first_eigenvector():
    from rotation.utils import first_eigenvector
    matrix = np.array([[2.0,0.0],[0.0,1.0]])
    first_eigen = first_eigenvector(matrix)
    npt.assert_almost_equal(np.abs(first_eigen),np.array([1.0,0.0]),decimal=6)

def test_IC2brain():
    from rotation.utils import IC2brain
    import nibabel as nib
    spatial_maps_data = np.array(np.reshape(np.arange(16),(2,2,2,2)),dtype=np.float64)
    mean_activation = np.array([[1.0,0.0,],[0.0,-1.0]])

    # Construct from spatial_maps data to spatial maps Nifti1Image
    spatial_map = nib.Nifti1Image(spatial_maps_data,affine = np.eye(4))
    brain_map = IC2brain(spatial_map,mean_activation)
    brain_map_data = brain_map.get_fdata()

    brain_map_true = np.array(np.reshape(np.array([i * (-1) ** i for i in range(16)]),(2,2,2,2)),dtype=np.float64)
    npt.assert_equal(brain_map_data,brain_map_true)

def test_IC2surface():
    from rotation.utils import IC2surface
    import nibabel as nib

    spatial_map_data = np.array([[0,2,4,6,8,10,12,14],[1,3,5,7,9,11,13,15]], dtype=np.float64)
    mean_activation = np.array([[1.0, 0.0, ], [0.0, -1.0]])
    axis_1 = nib.cifti2.cifti2_axes.ScalarAxis([f'Component {i + 1}' for i in range(2)])
    #axis_2 = nib.cifti2.cifti2_axes.BrainModelAxis(['CORTEX_LEFT'] * 2,vertex=np.arange(2)+1,affine=np.eye(4),volume_shape=(1,1,1))
    axis_2 = nib.cifti2.cifti2_axes.BrainModelAxis.from_mask(np.ones((2, 2, 2)),affine=np.eye(4),name='thalamus_left')
    header = nib.cifti2.cifti2.Cifti2Header.from_axes((axis_1, axis_2))
    spatial_map = nib.cifti2.cifti2.Cifti2Image(spatial_map_data, header)

    surface_map = IC2surface(spatial_map, mean_activation)
    surface_map_data = surface_map.get_fdata()

    surface_map_true = np.array([[0,2,4,6,8,10,12,14],[-1,-3,-5,-7,-9,-11,-13,-15]], dtype=np.float64)
    npt.assert_equal(surface_map_data, surface_map_true)



def test_pairwise_fisher_z_correlations():
    from rotation.utils import pairwise_fisher_z_correlations
    x1, x2, x3 = -0.2,0.5,0.3
    y1, y2, y3 = 0.7,-0.1,0.4

    def fisher_z_inverse(x):
        return (np.exp(2 * x) - 1) /  (np.exp(2 * x) + 1)

    answer = np.corrcoef(np.array([[x1,x2,x3],[y1,y2,y3]]))


    matrices = np.array([[[0.0,x1,x2],[x1,0.0,x3],[x2,x3,0.0]],
                         [[0.0,y1,y2],[y1,0.0,y3],[y2,y3,0.0]]])
    matrices = fisher_z_inverse(matrices) + np.eye(3)

    npt.assert_almost_equal(pairwise_fisher_z_correlations(matrices),answer,decimal=6)

def test_group_high_pass_filter():

    from rotation.utils import group_high_pass_filter
    N = 1200 # Number of time points
    T = 0.7 # Time resolution (seconds)
    fs = 1 / T # Sampling frequency


    f1 = 0.2
    f2 = 0.3
    t = np.array([i * T for i in range(N)])
    signal_1 = np.sin(2 * np.pi * f1 * t)
    signal_2 = np.cos(2 * np.pi * f2 * t)
    signals = [np.array([signal_1,signal_2]).T]

    filtered_signals = group_high_pass_filter(signals)[0].T
    filtered_signal_1 = filtered_signals[0]
    filtered_signal_2 = filtered_signals[1]
    
    frequencies = np.fft.fftfreq(N,1/fs)
    spectrum_1 = np.fft.fft(filtered_signal_1,norm='forward')
    spectrum_2 = np.fft.fft(filtered_signal_2,norm='forward')
    '''
    import matplotlib.pyplot as plt
    plt.plot(t,signal_1)
    plt.title('Original Signal 1 (t)')
    plt.show()
    plt.plot(frequencies,np.abs(np.fft.fft(signal_1,norm='forward')))
    plt.title('Original Signal 1 (f)')
    plt.show()
    plt.plot(t,signal_2)
    plt.title('Original Signal 2 (t)')
    plt.show()
    plt.plot(frequencies,np.abs(np.fft.fft(signal_2,norm='forward')))
    plt.title('Original Signal 2 (f)')
    plt.show()

    plt.plot(frequencies,np.abs(spectrum_1))
    plt.title('Filtered Signal 1 (f)')
    plt.show()
    plt.plot(frequencies,np.abs(spectrum_2))
    plt.title('Filtered Signal 2 (f)')
    plt.show()
    '''
    npt.assert_almost_equal(np.max(np.abs(spectrum_1)),0,decimal=3)
    npt.assert_almost_equal(np.max(np.abs(spectrum_2)),0.5,decimal=2)

def test_regularisation():
    from rotation.utils import regularisation

    matrix = np.eye(2)
    regularised_matrix = regularisation(matrix, 1e-6)
    npt.assert_equal(regularised_matrix,np.eye(2) * (1 + 1e-6))

    matrices = np.array([[[1.0,0.0],[0.0,1.0]],[[1.0,0.0],[0.0,1.0]]])
    regularised_matrix = regularisation(matrices,1e-6)
    npt.assert_equal(regularised_matrix,np.array([[[1.0,0.0],[0.0,1.0]],[[1.0,0.0],[0.0,1.0]]])*(1 + 1e-6))