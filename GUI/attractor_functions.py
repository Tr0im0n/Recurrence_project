import numpy as np
import matplotlib.pyplot as plt
def lorenz(xyz, *, s=10, r=28, b=2.667):
    x, y, z = xyz
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return np.array([x_dot, y_dot, z_dot])

def chua(xyz, *, alpha=15.6, beta=28, m0=-1.143, m1=-0.714):
    x, y, z = xyz
    h = m1 * x + 0.5 * (m0 - m1) * (abs(x + 1) - abs(x - 1))
    x_dot = alpha * (y - x - h)
    y_dot = x - y + z
    z_dot = -beta * y
    return np.array([x_dot, y_dot, z_dot])

def rossler(xyz, *, a=0.2, b=0.2, c=5.7):
    x, y, z = xyz
    x_dot = -y - z
    y_dot = x + a * y
    z_dot = b + z * (x - c)
    return np.array([x_dot, y_dot, z_dot])


def chen(xyz, *, a=40, b=3, c=28):
    x, y, z = xyz
    x_dot = a * (y - x)
    y_dot = (c - a) * x - x * z + c * y
    z_dot = x * y - b * z
    return np.array([x_dot, y_dot, z_dot])



