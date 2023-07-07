import numpy as np
import numpy.testing as npt

def test_prepare_data():
    import shutil, os, pathlib
    from rotation.preprocessing import PrepareData
    temp_dir = './test_temp/'
    # Create the directory if not exists
    if not os.path.exists(temp_dir):
        print(f'Create the temporary directory {temp_dir}')
        os.makedirs(temp_dir)
    
    # Create a subject
    subj_name = '10001'
    data = np.array([[-1,1,-1,1],[1,-1,1,-1]]).T
    np.savetxt(f'{temp_dir}{subj_name}.txt', data)
    
    # Use the PrepareData class
    prepare_data = PrepareData(pathlib.Path(temp_dir),2)
    subj,result = prepare_data.load()
    npt.assert_equal(subj[0],'10001')
    npt.assert_equal(result[0],np.array([[-1,1],[1,-1]]))
    npt.assert_equal(result[1],np.array([[-1,1],[1,-1]]))
    
    # Delete the directory
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f'Delete temporary directory {temp_dir}')

def test_z_score():
    
    # Example 1: one session
    x = np.array([[0.5,1.5,2.5],[-2.0,-1.0,0.0]]).T
    y = z_score(x)
    npt.assert_equal(y[:,0], y[:,1])
    
    # Example 2: two sessions
    np.random.seed(42)
    x1 = np.random.normal(loc=100.0,scale=100.0,size=(100,2))
    x2 = np.random.normal(loc=0.0,scale=2.0,size=(100,2))
    x = np.concatenate([x1,x2])
    y = z_score(x,n_session=2)
    
    # Check the data are z-scored
    npt.assert_almost_equal(np.mean(y[:100,:]),0.0,decimal=3)
    npt.assert_almost_equal(np.std(y[:100,:]),1.0,decimal=3)
    npt.assert_almost_equal(np.mean(y[100:,:]),0.0,decimal=3)
    npt.assert_almost_equal(np.std(y[100:,:]),1.0,decimal=3)
    