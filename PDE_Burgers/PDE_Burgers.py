import os

import numpy as np
import scipy.io as reader

abs_dir= os.path.dirname(__file__) + '/'
PDE_datafile = abs_dir+'burgers_shock.mat'
PDE_dim = 2
PDE_vars = ['x', 't']
PDE_scale = {
    'x': (-1, 1),
    't': (0, 1)
}
PDE_analytic_solution = False
PDE_description = 'Ut + U * Ux = (0.01/pi) * Uxx'
PDE_initial_condition = ['PDE_ic1(x, t=0)']
PDE_boundary_condition = ['PDE_bc1(t, x=-1)', 'PDE_bc2(t,x=1)']


def PDE_ic1(x, t=0):
    return -np.sin(np.pi * x)


def PDE_bc1(t, x=-1):
    return t - t


def PDE_bc2(t, x=1):
    return t - t


def PDE_get_data():
    data = reader.loadmat(PDE_datafile)
    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = np.real(data['usol']).T
    X, T = np.meshgrid(x, t)

    return {
        'data': data,
        'x': x,
        'x_dim': len(x),
        't': t,
        't_dim': len(t),
        'solution_type': 'X T U',
        'solution': [X, T, Exact]
    }