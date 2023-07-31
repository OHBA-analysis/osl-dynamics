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

def test_first_eigenvector():
    from rotation.utils import first_eigenvector
    matrix = np.array([[2.0,0.0],[0.0,1.0]])
    first_eigen = first_eigenvector(matrix)
    npt.assert_equal(first_eigen,np.array([1.0,0.0]))

def test_IC2brain():
    from rotation.utils import IC2brain
    import nibabel as nib
    spatial_maps_data = np.reshape(np.arange(16),(2,2,2,2))
    mean_activation = np.array([[1.0,0.0,],[0.0,-1.0]])

    # Construct from spatial_maps data to spatial maps Nifti1Image
    spatial_map = nib.Nifti1Image(spatial_maps_data,affine = np.eye(4))
    brain_map = IC2brain(spatial_map,mean_activation)
    brain_map_data = brain_map.get_fdata()

    brain_map_true = np.reshape(np.array([i * (-1) ** i for i in range(16)]),(2,2,2,2))
    np.assert_equal(brain_map_data,brain_map_true)