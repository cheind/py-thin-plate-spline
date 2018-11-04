# Copyright 2018 Christoph Heindl.
#
# Licensed under MIT License
# ============================================================

import numpy as np

class TPS:       
    @staticmethod
    def fit(c, lambd=0.):        
        n = c.shape[0]

        U = TPS.u(TPS.d(c, c))
        K = U + np.eye(n, dtype=np.float32)*lambd

        P = np.ones((n, 3), dtype=np.float32)
        P[:, 1:] = c[:, :2]

        v = np.zeros(n+3, dtype=np.float32)
        v[:n] = c[:, -1]

        A = np.zeros((n+3, n+3), dtype=np.float32)
        A[:n, :n] = K
        A[:n, -3:] = P
        A[-3:, :n] = P.T

        theta = np.linalg.solve(A, v) # p has structure w,a
        return theta
        
    @staticmethod
    def d(a, b):
        return np.sqrt(np.square(a[:, None, :2] - b[None, :, :2]).sum(-1))

    @staticmethod
    def u(r):
        return r**2 * np.log(r + 1e-6)

    @staticmethod
    def z(x, c, theta):
        x = np.atleast_2d(x)
        U = TPS.u(TPS.d(x, c))
        b = np.dot(U, theta[:-3])
        return theta[-3] + theta[-2]*x[:, 0] + theta[-1]*x[:, 1] + b

def normalized_grid(shape):
    X = np.linspace(0, 1, shape[1], dtype=np.float32) # note: for torch this needs to be changed
    Y = np.linspace(0, 1, shape[0], dtype=np.float32)
    X, Y = np.meshgrid(X, Y)
    xy = np.hstack((X.reshape(-1, 1),Y.reshape(-1, 1)))
    return X, Y, xy

def compute_densegrid(c_src, c_dst, dshape):    
    delta = c_src - c_dst
    
    cx = np.column_stack((c_dst, delta[:, 0]))
    cy = np.column_stack((c_dst, delta[:, 1]))
        
    X, Y, xy = normalized_grid(dshape)

    theta_dx = TPS.fit(cx)
    theta_dy = TPS.fit(cy)
    
    dx = TPS.z(xy, c_dst, theta_dx).reshape(dshape[:2])
    dy = TPS.z(xy, c_dst, theta_dy).reshape(dshape[:2])

    map_x = (X + dx).astype('float32')
    map_y = (Y + dy).astype('float32')
    grid = np.stack((map_x, map_y), -1)
    
    return grid # H'xW'x2 grid[i,j] in range [0..1]

def densegrid_to_remap(grid, sshape):
    '''Convert a dense grid to OpenCV's remap compatible maps.
    
    Params
    ------
    grid : HxWx2 array
        Normalized flow field coordinates as computed by compute_densegrid.
    sshape : tuple
        Height and width of source image in pixels.


    Returns
    -------
    mapx : HxW array
    mapy : HxW array
    '''

    return grid[:, :, 0] * sshape[1], grid[:, :, 1] * sshape[0]


def compute_densegrid_from_theta(c_dst, theta_dx, theta_dy, dshape):    
    X, Y, xy = normalized_grid(dshape)

    dx = TPS.z(xy, c_dst, theta_dx).reshape(dshape[:2])
    dy = TPS.z(xy, c_dst, theta_dy).reshape(dshape[:2])

    map_x = (X + dx).astype('float32')
    map_y = (Y + dy).astype('float32')
    grid = np.stack((map_x, map_y), -1)
    
    return grid # H'xW'x2 grid[i,j] in range [0..1]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    c = np.array([
        [0., 0, 0.5],
        [1., 0, 0.0],
        [1., 1, 0.0],
        [0, 1, 0.0],
    ])

    tps = TPS()
    tps.fit(c)

    X = np.linspace(0, 1, 10)
    Y = np.linspace(0, 1, 10)
    X, Y = np.meshgrid(X, Y)
    xy = np.hstack((X.reshape(-1, 1),Y.reshape(-1, 1)))
    Z = tps(xy).reshape(10, 10)

    fig, ax = plt.subplots()
    c = ax.contour(X, Y, Z)
    ax.clabel(c, inline=1, fontsize=10)
    plt.show()    