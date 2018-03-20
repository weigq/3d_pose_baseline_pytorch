from __future__ import division

import h5py
import numpy as np


def project_point_radial(P, R, T, f, c, k, p):
    """
    Args
    P: Nx3 points in world coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
    f: 2x1 (scalar) Camera focal length
    c: 2x1 Camera center
    k: 3x1 Camera radial distortion coefficients
    p: 2x1 Camera tangential distortion coefficients
    Returns
    Proj: Nx2 points in pixel space
    D: 1xN depth of each point in camera space
    radial: 1xN radial distortion per point
    tan: 1xN tangential distortion per point
    r2: 1xN squared radius of the projected points before distortion
    """

    # P is a matrix of 3-dimensional points
    assert len(P.shape) == 2
    assert P.shape[1] == 3

    N = P.shape[0]
    X = R.dot(P.T - T)  # rotate and translate
    XX = X[:2, :] / X[2, :]  # 2x16
    r2 = XX[0, :] ** 2 + XX[1, :] ** 2  # 16,

    radial = 1 + np.einsum('ij,ij->j', np.tile(k, (1, N)), np.array([r2, r2 ** 2, r2 ** 3]))  # 16,
    tan = p[0] * XX[1, :] + p[1] * XX[0, :]  # 16,

    tm = np.outer(np.array([p[1], p[0]]).reshape(-1), r2)  # 2x16

    XXX = XX * np.tile(radial + tan, (2, 1)) + tm  # 2x16

    Proj = (f * XXX) + c  # 2x16
    Proj = Proj.T

    D = X[2, ]

    return Proj, D, radial, tan, r2


def world_to_camera_frame(P, R, T):
    """
    :param P: Nx3 points in world coords
    :param R: 3x3 camera rotation matrix
    :param T: 3x1 camera translation params
    :return: X_cam: Nx3 3d points in camera coords
    """

    assert len(P.shape) == 2
    assert P.shape[1] == 3

    X_cam = R.dot(P.T - T)  # rotate and translate

    return X_cam.T


def camera_to_world_frame(P, R, T):
    """Inverse of world_to_camera_frame

  Args
    P: Nx3 points in camera coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
  Returns
    X_cam: Nx3 points in world coordinates
  """

    assert len(P.shape) == 2
    assert P.shape[1] == 3

    X_cam = R.T.dot(P.T) + T  # rotate and translate

    return X_cam.T


def load_camera_params(hf, path):
    """
    load camera paprameters
    :param hf:
    :param path: keys in hf
    :return:
        R: 3x3 cam rotation matric
        T: 3x1 cam translation param
        f:
    """
    R = hf[path.format('R')][:]
    R = R.T

    T = hf[path.format('T')][:]
    f = hf[path.format('f')][:]
    c = hf[path.format('c')][:]
    k = hf[path.format('k')][:]
    p = hf[path.format('p')][:]

    name = hf[path.format('Name')][:]
    name = "".join([chr(item) for item in name])

    return R, T, f, c, k, p, name


def load_cameras(bpath='cameras.h5', subjects=None):
    """
    :param bpath: *.h5
    :param subjects:
    :return: (dict)
    """

    if subjects is None:
        subjects = [1, 5, 6, 7, 8, 9, 11]
    rcams = {}

    with h5py.File(bpath, 'r') as hf:
        for s in subjects:
            for c in range(4):  # There are 4 cameras in human3.6m
                a = load_camera_params(hf, 'subject%d/camera%d/{0}' % (s, c + 1))
                rcams[(s, c + 1)] = a

    return rcams
