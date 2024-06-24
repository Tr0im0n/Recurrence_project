import numpy as np

# Must have a function named 'function' which takes
def function(xyz, *, s=10, r=28, b=2.667):
    '''
    :param xyz: np.array with last x, y, and z coordinates
    :param s, r, b: constants
    :return: np.array with new x, y, x coordinates
    '''
    x, y, z = xyz
    # x_dot = y+x
    # y_dot = r+x
    # z_dot = z
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return np.array([x_dot, y_dot, z_dot])

