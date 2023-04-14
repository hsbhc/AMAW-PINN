import os

import numpy as np
import scipy.io as reader

abs_dir = os.path.dirname(__file__) + '/'

PDE_datafile_u = abs_dir + 'u.csv'
PDE_datafile_v = abs_dir + 'v.csv'

PDE_dim = 2
PDE_vars = ['x', 'y']
PDE_scale = {
    'x': (0, 1),
    'y': (0, 1),
}
PDE_analytic_solution = False
PDE_description = 'UUx+VUy+Px-1/Re (Uxx+Uyy)=0 \n ' \
                  'UVx+VVy+Px-1/Re (Vxx+Vyy)=0  \n' \
                  'Ux+Vy=0'
PDE_initial_condition = []
PDE_boundary_condition = ['PDE_bc1(x, y=1)', 'PDE_bc2(x, y) on x=0 or x=1 or y=0']

Re = 100


def PDE_bc1(x, y=1):
    x = np.expand_dims(x, axis=1)
    return np.concatenate((np.ones_like(x), np.zeros_like(x)), axis=1)


def PDE_bc2(x, y):
    x = np.expand_dims(x, axis=1)
    return np.concatenate((np.zeros_like(x), np.zeros_like(x)), axis=1)


def PDE_get_data():
    nx = 100
    ny = 100
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y)

    u_ref = np.genfromtxt(PDE_datafile_u, delimiter=',')
    v_ref = np.genfromtxt(PDE_datafile_v, delimiter=',')
    velocity_ref = np.sqrt(u_ref ** 2 + v_ref ** 2)

    return {
        'data': [u_ref, v_ref],
        'x': x,
        'x_dim': len(x),
        'y': y,
        'y_dim': len(y),
        'solution_type': 'X Y U V |result|',
        'solution': [X.T, Y.T, u_ref, v_ref, velocity_ref]
    }
