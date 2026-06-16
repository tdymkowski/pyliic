#! /usr/bin/env python3
import numpy as np
import sys

from pyliic.eckart import apply_eckart
from .fileio import write_xyz_traj
from .gmatrix import *
from .data import ANG2AU
from .interpolation import liic, make_distance_dihedral_grid
from .utils import XYZ
from .geometry_generator import LIICGeometryGenerator


def main():
    n_images = 100
    xyzpath = "/home/tdymkowski/Documents/projects/liic/s1_geometries.xyz"
    xyz_lst = XYZ.read_xyz(xyzpath)

    indices = [2, 1, 10, 11]
    dihedral_indices = [4, 3, 9, 2]

    react = xyz_lst[0]
    masses = react.get_masses_au()
    prod = xyz_lst[1]
    order = [9, 11, 8, 0, 1, 2, 3, 4, 5, 6, 7, 10, 12, 13, 14, 15]

    traj_pt = liic(react, prod, n_images=n_images, order=order)
    
    q1_r = np.array([atoms.get_distance(0, 1) for atoms in traj_pt])
    q2_r = np.array([atoms.get_distance(1, 2) for atoms in traj_pt])

    phi0 = traj_pt[0].get_dihedral(*dihedral_indices)
    phi1 = 360.
    make_geom = LIICGeometryGenerator(react, prod, proton_idx=11, atom1_idx=9, dihedral_indices=[1, 0, 6, 8], moving_indices=[8, 7, 10, 11])
    q1_fine = np.linspace(np.min(q1_r), np.max(q1_r), 200)
    q2_fine = np.linspace(0., 360., 200)
    geoms = []
    for q1_f in q1_fine:
        for q2_f in q2_fine:
            geom_q1 = make_geom(q1_f, phi=q2_f)
            geoms.append(geom_q1)
    geoms = apply_eckart(geoms)
    write_xyz_traj("traj_liic.xyz", geoms)

    q1_label = r"$r{O1H}$"
    positions = np.array([g.get_positions() for g in geoms]) * ANG2AU
    n_q1 = len(q1_fine)
    n_q2 = len(q2_fine)
    n_atoms = positions.shape[1]
    positions = positions.reshape(n_q1, n_q2, n_atoms, 3)
    q1_fine *= ANG2AU
    q2_fine_rad = np.deg2rad(q2_fine)
    G = get_G_matrix(q1_fine, q2_fine_rad, positions, masses, plot=True, save_op=False, method="fd", q1_label=q1_label)
#    phi_grid_deg = np.linspace(phi0, phi1, n_images)
#    traj_grid = make_distance_dihedral_grid(traj_r=traj_pt,
#                                            dihedral_indices=dihedral_indices,
#                                            moving_indices=indices,
#                                            phi_grid_deg=phi_grid_deg)
#    positions = np.array([[atoms.get_positions() for atoms in row] for row in traj_grid])
##    q_r = np.linspace(0, 1, len(q1_r))
##    q1_label = r"t"
##    q_r = q2_r - q1_r
#    q_r = q1_r
#    q1_label = r"$r{O1H}$"
##    q_r = q2_r
##    q1_label = r"$r{O2H}$"
#    q_phi = np.unwrap(np.deg2rad(phi_grid_deg))
#
#    positions = positions * ANG2AU
##    q_r = q_r  * ANG2AU
#
#    G = get_G_matrix(q_r, q_phi, positions, masses, plot=True, save_op=False, method="fd", q1_label=q1_label)
#    write_xyz_traj("liic_path_pt_rot.xyz", traj_grid.flatten())
