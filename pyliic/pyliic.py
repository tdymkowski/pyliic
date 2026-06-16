#! /usr/bin/env python3
import numpy as np
import sys
from .utils import XYZ, ZMatrix
from .fileio import write_xyz_traj
from .gmatrix import *
from .data import ANG2AU
from .eckart import apply_eckart


def interpolate_angle(a0, a1, t):
    return (1. - t) * a0 + t * a1


def interpolate_dihedral(phi0, phi1, t):
    dphi = (phi1 - phi0 + 180.) % 360. - 180.
    phi = phi0 + t * dphi
    return phi % 360.


def interpolate_bond(r0, r1, t):
    return (1. - t) * r0 + t * r1


def liic(atoms0, atoms1, n_images=101, order=None):
#    atoms0 = atoms0.copy()
#    atoms1 = atoms1.copy()
    if order is not None:
        atoms0 = atoms0.reorder_atoms(order)
        atoms1 = atoms1.reorder_atoms(order)
    zmat0 = atoms0.to_zmat()
    zmat1 = atoms1.to_zmat()

    traj = []
    zmats = []
    for t in np.linspace(0, 1, n_images):
        bonds = [None] * zmat0.n_atoms
        angels = [None] * zmat0.n_atoms
        dihedrals = [None] * zmat0.n_atoms
        for i in range(zmat0.n_atoms):
            if zmat0.bonds[i] is not None:
                bonds[i] = interpolate_bond(zmat0.bonds[i], zmat1.bonds[i], t)
            if zmat0.angles[i] is not None:
                angels[i] = interpolate_angle(zmat0.angles[i], zmat1.angles[i], t)
            if zmat0.dihedrals[i] is not None:
                dihedrals[i] = interpolate_dihedral(zmat0.dihedrals[i], zmat1.dihedrals[i], t)
        zmat_t = ZMatrix(symbols=zmat0.symbols,
                         bonds=bonds,
                         bond_refs=zmat0.bond_refs,
                         angles=angels,
                         angle_refs=zmat0.angle_refs,
                         dihedrals=dihedrals,
                         dihedral_refs=zmat0.dihedral_refs)
        zmats.append(zmat_t)
        xyz = zmat_t.to_xyz()
        traj.append(xyz)

    traj = apply_eckart(traj, ref_idx=len(traj) // 2)
    return traj


def interpolate_dihedral_rotation(atoms, dihedral_indices, final_angle, moving_indices, n_images=101, order=None):
    atoms0 = atoms.copy()
    if order is not None:
        atoms0 = atoms.reorder_atoms(order)
    phi0 = atoms0.get_dihedral(*dihedral_indices)
    phi1 = final_angle
    dphi = (phi1 - phi0 + 180.) % 360. - 180.

    traj = []
    for t in np.linspace(0, 1, n_images):
        phi_t = (phi0 + t * dphi) % 360.
        atoms_t = atoms0.copy()
        atoms_t.set_dihedral(*dihedral_indices, phi_t, indices=moving_indices)
        traj.append(atoms_t)
    return traj


def make_distance_dihedral_grid(traj_r, dihedral_indices, moving_indices, phi_grid_deg, scaffold=None):

    grid = []

    for base_geom in traj_r:
        row = []
        for phi in phi_grid_deg:
            atoms = base_geom.copy()
            atoms.set_dihedral(*dihedral_indices, phi, indices=moving_indices)
            row.append(atoms)
        grid.append(row)

    n_r = len(traj_r)
    n_phi = len(phi_grid_deg)
    flat_grid = [geom for row in grid for geom in row]
    flat_grid = apply_eckart(flat_grid,
                             ref_idx=len(flat_grid) // 2,
                             scaffold=scaffold)
    grid = np.array(flat_grid).reshape(n_r, n_phi)

    return np.array(grid, dtype=object)


def remap_indices_after_reorder(indices, order):
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(order)}
    return [old_to_new[i] for i in indices]


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

    phi_grid_deg = np.linspace(phi0, phi1, n_images)
    traj_grid = make_distance_dihedral_grid(traj_r=traj_pt,
                                            dihedral_indices=dihedral_indices,
                                            moving_indices=indices,
                                            phi_grid_deg=phi_grid_deg)
    positions = np.array([[atoms.get_positions() for atoms in row] for row in traj_grid])
    q_r = q2_r - q1_r
#    q_r = q1_r
#    q1_label = r"$r{O1H}$"
#    q_r = q2_r
#    q1_label = r"$r{O2H}$"
    q_phi = np.unwrap(np.deg2rad(phi_grid_deg))

    positions = positions * ANG2AU
    q_r = q_r * ANG2AU
    G = get_G_matrix(q_r, q_phi, positions, masses, plot=True, save_op=True, method="fd")

    write_xyz_traj("liic_path_pt_rot.xyz", traj_grid.flatten())
