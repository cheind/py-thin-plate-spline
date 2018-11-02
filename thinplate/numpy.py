# Copyright 2018 Christoph Heindl.
#
# Licensed under MIT License
# ============================================================

import numpy as np

class TPS:
    def __init__(self):
        self.p = None
        
    def fit(self, c, lambd=0.):        
        n = c.shape[0]        
        U = self._u(self._d(c, c))
        K = U + np.eye(n)*lambd

        P = np.ones((n, 3))
        P[:, 1:] = c[:, :2]

        v = np.zeros(n+3)
        v[:n] = c[:, -1]

        A = np.zeros((n+3, n+3))
        A[:n, :n] = K
        A[:n, -3:] = P
        A[-3:, :n] = P.T

        self.p = np.linalg.solve(A, v) # p has structure w,a
        self.c = np.copy(c)
        
    def __call__(self, x):
        x = np.atleast_2d(x)
        U = self._u(self._d(x, self.c))
        b = np.dot(U, self.p[:-3])
        return self.p[-3] + self.p[-2]*x[:, 0] + self.p[-1]*x[:, 1] + b
        
    def _d(self, a, b):
        return np.sqrt(np.square(a[:, None, :2] - b[None, :, :2]).sum(-1))
    
    def _u(self, r):
        return r**2 * np.log(r + 1e-6)

def compute_densegrid(img, c_src, c_dst, dshape=None):
    sshape = img.shape
    if dshape is None:
        dshape = sshape
    
    delta = c_src - c_dst
    
    cx = np.column_stack((c_dst, delta[:, 0]))
    cy = np.column_stack((c_dst, delta[:, 1]))
    
    tps_x, tps_y = TPS(), TPS()
    tps_x.fit(cx)
    tps_y.fit(cy)
    
    X = np.linspace(0, 1, dshape[1]) # note: for torch this needs to be changed
    Y = np.linspace(0, 1, dshape[0])
    X, Y = np.meshgrid(X, Y)
    xy = np.hstack((X.reshape(-1, 1),Y.reshape(-1, 1)))

    map_x = (tps_x(xy).reshape(dshape[:2]) + X).astype('float32')
    map_y = (tps_y(xy).reshape(dshape[:2]) + Y).astype('float32')
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