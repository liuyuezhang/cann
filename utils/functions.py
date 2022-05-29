import numpy as np
import cupy as cp


def relu(x):
    return x * (x > 0)


def l2_distance(x, y, d=None):
    # d is None: line, equiv to perimeter is np.inf
    xp = cp.get_array_module(x)
    s = distance(x, y, d=d)
    res = xp.sqrt(xp.sum(s**2, -1))
    return res.astype(xp.float32)


def dog(d, a1=1., s1=1.05 * 3. / (13.**2), a2=1., s2=1. * 3. / (13.**2)):
    xp = cp.get_array_module(d)
    return a1 * xp.exp(-s1 * d**2) - a2 * xp.exp(-s2 * d**2)


def dog_grad(d, x, a1=1., s1=1.05 * 3. / (13.**2), a2=1., s2=1. * 3. / (13.**2)):
    xp = cp.get_array_module(d)
    return (a1 * s1 * xp.exp(-s1 * d**2) - a2 * s2 * xp.exp(-s2 * d**2)) * (-2 * x)


def subtract(x, y, d=None):
    """
    subtraction on a ring is adding its inverse element (group theory)
    it has nothing to do with the absolute value, since everything is ``relative'' quantity in convolution
    """
    xp = cp.get_array_module(x)
    return x - y if d is None else xp.mod(x + (d/2 - y), d) - d/2


def distance(x, y, d=None):
    xp = cp.get_array_module(x)
    abs_val = xp.abs(x - y)
    if d is None:
        res = abs_val
    else:
        abs_val = abs_val % d
        res = xp.minimum(abs_val, d - abs_val)
    return res.astype(xp.float32)


def get_D(ng=64, l=0, e=None, require_Dx=False):
    """
    Define a ring on (-ng/2, ..., 0, ...,  ng/2-1) with length ng
    Calculate the relative x, y coordinates matrix Dx, Dy, and the relative distance matrix D
    """
    a, b, d = -ng/2, ng/2-1, ng
    x = np.linspace(a, b, d)
    y = np.linspace(a, b, d)
    x, y = np.meshgrid(x, y, indexing='xy')  # or 'ij'?
    x = x.reshape(ng * ng, -1)
    y = y.reshape(ng * ng, -1)
    Dx = subtract(x, x.T, d=d)
    Dy = subtract(y, y.T, d=d)

    if e is None:
        dx = distance(x, x.T, d=d)
        dy = distance(y, y.T, d=d)
    else:
        ex, ey = e
        dx = distance(x, (x - l * ex).T, d=d)
        dy = distance(y, (y - l * ey).T, d=d)
    D = np.sqrt(dx ** 2 + dy ** 2)
    return D if not require_Dx else (D, Dx, Dy)