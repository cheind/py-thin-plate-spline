import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import thinplate as tps

from numpy.testing import assert_allclose

def test_pytorch_grid():

    c_dst = np.array([
        [0., 0],
        [1., 0],    
        [1, 1],
        [0, 1],  
    ], dtype=np.float32)


    c_src = np.array([
        [10., 10],
        [20., 10],    
        [20, 20],
        [10, 20],  
    ], dtype=np.float32) / 40.

    np_grid, ndx, ndy = tps.tps_grid(c_src, c_dst, (20,20), return_theta=True)
    theta = torch.tensor(np.stack((ndx, ndy), -1)).unsqueeze(0)
    pth_grid = tps.torch.tps_grid(theta, torch.tensor(c_dst), (1, 1, 20, 20)).squeeze().numpy()
    pth_grid = (pth_grid + 1) / 2 # convert [-1,1] range to [0,1]

    assert_allclose(np_grid, pth_grid)