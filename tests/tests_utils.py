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