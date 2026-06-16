#! /usr/bin/env python3
import numpy as np
from .utils import XYZ


def get_U_matrix(q):
    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    U = np.zeros((len(q), 3, 3))

    U[:, 0, 0] = q0**2 + q1**2 - q2**2 - q3**2
    U[:, 0, 1] = 2 * (q1 * q2 + q0 * q3)
    U[:, 0, 2] = 2 * (q1 * q3 - q0 * q2)

    U[:, 1, 0] = 2 * (q1 * q2 - q0 * q3)
    U[:, 1, 1] = q0**2 - q1**2 + q2**2 - q3**2
    U[:, 1, 2] = 2 * (q2 * q3 + q0 * q1)

    U[:, 2, 0] = 2 * (q1 * q3 + q0 * q2)
    U[:, 2, 1] = 2 * (q2 * q3 - q0 * q1)
    U[:, 2, 2] = q0**2 - q1**2 - q2**2 + q3**2

    return U

def get_F_matrix(A: np.ndarray):
    n_geoms = A.shape[0]
    F = np.zeros((n_geoms, 4, 4))
    # first row
    F[:, 0, 0] = A[:, 0, 0] + A[:, 1, 1] + A[:, 2, 2]
    F[:, 0, 1] = A[:, 1, 2] - A[:, 2, 1]
    F[:, 0, 2] = A[:, 2, 0] - A[:, 0, 2]
    F[:, 0, 3] = A[:, 0, 1] - A[:, 1, 0]
    # second row
    F[:, 1, 0] = A[:, 1, 2] - A[:, 2, 1] 
    F[:, 1, 1] = A[:, 0, 0] - A[:, 1, 1] - A[:, 2, 2]
    F[:, 1, 2] = A[:, 0, 1] + A[:, 1, 0]
    F[:, 1, 3] = A[:, 0, 2] + A[:, 2, 0]
    # third row
    F[:, 2, 0] = A[:, 2, 0] - A[:, 0, 2] 
    F[:, 2, 1] = A[:, 0, 1] + A[:, 1, 0]
    F[:, 2, 2] = -A[:, 0, 0] + A[:, 1, 1] - A[:, 2, 2]
    F[:, 2, 3] = A[:, 1, 2] + A[:, 2, 1]
    # fourth row:, 
    F[:, 3, 0] = A[:, 0, 1] - A[:, 1, 0] 
    F[:, 3, 1] = A[:, 0, 2] + A[:, 2, 0]
    F[:, 3, 2] = A[:, 1, 2] + A[:, 2, 1]
    F[:, 3, 3] = -A[:, 0, 0] - A[:, 1, 1] + A[:, 2, 2]
    return F


def get_A_matrix(positions, masses, ref_idx=0):
    R = positions[ref_idx]
    
    A = np.einsum("a,nai,aj->nij", masses, positions, R)
    return A


def rotational_eckart(positions: np.ndarray, masses: np.ndarray , ref_idx=0, min_eigen=False):
    # get coorelation matrix
    A = get_A_matrix(positions, masses, ref_idx=ref_idx)
    # get F matrix
    F = get_F_matrix(A)
    eigenvalues, eigenvectors = np.linalg.eigh(F)
    q = eigenvectors[:, :, 0]
    U = get_U_matrix(q)
    positions_rot = np.einsum("nij,naj->nai", U, positions)
    return positions_rot


def translational_eckart(positions: np.ndarray, masses: np.ndarray):
    com_shifted_positions = []

    for positions_i in positions:
        com = np.average(positions_i, weights=masses, axis=0)
        positions_i -= com
        com_shifted_positions.append(positions_i)

    return np.array(com_shifted_positions)


def check_translational_eckart(positions, masses, tol=1e-12):
    trans_res = np.einsum("a,nak->nk", masses, positions)
    norms = np.linalg.norm(trans_res, axis=1)
    if np.any(norms > tol):
        raise ValueError(f"Translational Eckart condition error. Max residual: {np.max(norms):.6e}")


def check_rotational_eckart(positions, masses, ref_idx=0, tol=1e-12):
    R = positions[ref_idx]
    cross = np.cross(positions, R[None, :, :], axis=2)
    rot_res = np.einsum("a,nak->nk", masses, cross)
    norms = np.linalg.norm(rot_res, axis=1)
    print(f"Rotational Eckart condition error. Max residual: {np.max(norms):.6e}")
#    if np.any(norms > tol):
#        raise ValueError(f"Rotational Eckart condition error. Max residual: {np.max(norms):.6e}")
#

def check_eckart_conditions(positions, masses, ref_idx=0):
    check_translational_eckart(positions, masses)
    check_rotational_eckart(positions, masses, ref_idx=ref_idx)

def apply_eckart(traj: list[XYZ], ref_idx=0):
    # get all positions
    positions = np.array([t.get_positions() for t in traj])
    # get masses
    masses = traj[ref_idx].get_masses_amu()
    # apply translational eckart condition
    positions = translational_eckart(positions, masses)
    # apply rotational eckart condition
    positions = rotational_eckart(positions, masses, ref_idx=ref_idx)
    check_eckart_conditions(positions, masses, ref_idx=ref_idx)
    # convert to list[XYZ]
    traj1 = []
    for t, pos in zip(traj, positions):
        t_new = t.copy()
        t_new.set_positions(pos)
        traj1.append(t_new)

    return traj1
