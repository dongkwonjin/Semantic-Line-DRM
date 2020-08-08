import numpy as np
import numpy.linalg as LA


def neurvps_convert(data):
    # data -> two points of lines
    axy = data[0]
    bxy = data[1]

    a0, a1 = np.array(axy[:2]), np.array(axy[2:])
    b0, b1 = np.array(bxy[:2]), np.array(bxy[2:])
    xy = intersect(a0, a1, b0, b1) - 0.5

    vpts = np.array([[xy[0] / 200 - 1, 1 - xy[1] / 200, 1]])
    vpts[0] /= LA.norm(vpts[0])

    return np.float32(vpts)


def intersect(a0, a1, b0, b1):
    c0 = ccw(a0, a1, b0)
    c1 = ccw(a0, a1, b1)
    d0 = ccw(b0, b1, a0)
    d1 = ccw(b0, b1, a1)
    if abs(d1 - d0) > abs(c1 - c0):
        return (a0 * d1 - a1 * d0) / (d1 - d0)
    else:
        return (b0 * c1 - b1 * c0) / (c1 - c0)

def ccw(c, a, b):
    a0 = a - c
    b0 = b - c
    return a0[0] * b0[1] - b0[0] * a0[1]

