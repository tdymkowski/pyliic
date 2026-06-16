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
    traj = apply_eckart(traj)
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

import numpy as np


def cartesian_interpolation(atoms0, atoms1, n_images=101, order=None, remove_com=False):
    """
    Linear interpolation in Cartesian coordinates.

    Parameters
    ----------
    atoms0, atoms1
        XYZ/Atoms-like objects with:
            copy()
            reorder_atoms(order)
            get_positions()
            set_positions(positions)

    n_images : int
        Number of interpolated structures, including endpoints.

    order : list[int] or None
        Optional atom reorder applied to both endpoints.

    remove_com : bool
        If True, subtract center of mass from each interpolated structure.
        Only use this if your object has get_masses() or get_masses_au().

    Returns
    -------
    traj : list
        List of interpolated structures.
    """

    atoms0 = atoms0.copy()
    atoms1 = atoms1.copy()

    if order is not None:
        atoms0 = atoms0.reorder_atoms(order)
        atoms1 = atoms1.reorder_atoms(order)

    x0 = np.asarray(atoms0.get_positions(), dtype=float)
    x1 = np.asarray(atoms1.get_positions(), dtype=float)

    if x0.shape != x1.shape:
        raise ValueError("atoms0 and atoms1 must have the same number of atoms.")

    traj = []

    for t in np.linspace(0.0, 1.0, n_images):
        xt = (1.0 - t) * x0 + t * x1

        atoms_t = atoms0.copy()
        atoms_t.set_positions(xt)

        if remove_com:
            try:
                masses = atoms_t.get_masses()
            except AttributeError:
                masses = atoms_t.get_masses_au()

            com = np.average(atoms_t.get_positions(), axis=0, weights=masses)
            atoms_t.set_positions(atoms_t.get_positions() - com)

        traj.append(atoms_t)

    return traj



def make_distance_dihedral_grid(traj_r, dihedral_indices, moving_indices, phi_grid_deg):

    grid = []

    for base_geom in traj_r:
        row = []

        for phi in phi_grid_deg:
            atoms = base_geom.copy()
            atoms.set_dihedral(*dihedral_indices, phi, indices=moving_indices)
            row.append(atoms)
        row = apply_eckart(row)
        grid.append(row)

    return np.array(grid)


def remap_indices_after_reorder(indices, order):
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(order)}
    return [old_to_new[i] for i in indices]



def build_rotate_then_liic_grid(
    react,
    prod,
    order,
    dihedral_indices,
    moving_indices,
    n_images=101,
    phi1=180.0,
):
    # Reorder once
    react_ord = react.reorder_atoms(order)
    prod_ord = prod.reorder_atoms(order)
#    prod_ord = prod
#    react_ord = react
    # Remap indices to reordered atom numbering
#    dihedral_indices = remap_indices_after_reorder(dihedral_indices_old, order)
#    moving_indices = remap_indices_after_reorder(moving_indices_old, order)

    print("dihedral_indices:", dihedral_indices)
    print("moving_indices:", moving_indices)

    # Absolute dihedral grid
    phi0 = prod_ord.get_dihedral(*dihedral_indices)
    phi_grid_deg = np.linspace(phi0, phi1, n_images)
    q_phi = np.unwrap(np.deg2rad(phi_grid_deg))

    # Rotate product first
    rotated_products = []

    for phi in phi_grid_deg:
        atoms = prod_ord.copy()
        atoms.set_dihedral(
            *dihedral_indices,
            phi,
            indices=moving_indices,
        )
        rotated_products.append(atoms)
    # Debug: verify product dihedral changed
    phis_after = np.array([
        atoms.get_dihedral(*dihedral_indices)
        for atoms in rotated_products
    ])

    print("rotated product phi min:", phis_after.min())
    print("rotated product phi max:", phis_after.max())
    print("first 5 product phis:", phis_after[:5])
    print("last 5 product phis:", phis_after[-5:])

    # Then LIIC reactant -> each rotated product
    traj_rot_pt = []

    for prod_rot in rotated_products:
        print(react_ord.to_zmat())
        print(prod_rot.to_zmat())
#        traj = liic(
#            react_ord,
#            prod_rot,
#            n_images=n_images,
#            order=None,
#        )
        traj = cartesian_interpolation(
            react_ord,
            prod_rot,
            n_images=n_images,
            order=None,
        )
        traj_rot_pt.append(traj)

    # positions[i, j] = LIIC index i, dihedral index j
    positions = np.array([
        [traj_rot_pt[j][i].get_positions() for j in range(n_images)]
        for i in range(n_images)
    ], dtype=float)

    # Natural coordinate for LIIC direction
    q_s = np.linspace(0.0, 1.0, n_images)

    # Optional: distance diagnostic
    all_r = np.array([
        [traj_rot_pt[j][i].get_distance(0, 1) for i in range(n_images)]
        for j in range(n_images)
    ], dtype=float)

    print("max r-grid deviation:", np.max(np.abs(all_r - all_r[0][None, :])))
    print("positions shape:", positions.shape)

    return traj_rot_pt, positions, q_s, q_phi, react_ord, prod_ord, dihedral_indices, moving_indices


def main():
    n_images = 101
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
    print(q2_r)

    phi0 = traj_pt[0].get_dihedral(*dihedral_indices)
#    phi0 = -60
    phi1 = 180.

    phi_grid_deg = np.linspace(phi0, phi1, n_images)
    traj_grid = make_distance_dihedral_grid(traj_r=traj_pt,
                                            dihedral_indices=dihedral_indices,
                                            moving_indices=indices,
                                            phi_grid_deg=phi_grid_deg)
    positions = np.array([[atoms.get_positions() for atoms in row] for row in traj_grid])
    q_r = q2_r - q1_r
#    q_r = q1_r
#    q_r = q2_r
    q_phi = np.unwrap(np.deg2rad(phi_grid_deg))

    positions = positions * ANG2AU
    q_r = q_r * ANG2AU
    G = get_G_matrix(q_r, q_phi, positions, masses, plot=True, method="fd")

    write_xyz_traj("liic_path_pt_rot.xyz", traj_grid.flatten())
#    react = xyz_lst[0]
#    prod = xyz_lst[1]
#    traj_rot_pt, positions, q_s, q_phi, react_ord, prod_ord, dihedral_indices, moving_indices = (
#        build_rotate_then_liic_grid(
#            react=react,
#            prod=prod,
#            order=order,
#            dihedral_indices=dihedral_indices,
#            moving_indices=indices,
#            n_images=n_images,
#            phi1=180.0,
#        )
#    )
#    positions_au = positions * ANG2AU
#    masses = react_ord.get_masses_au()
#
#    traj_grid_flat = []
#
#    for i in range(n_images):
#        for j in range(n_images):
#            atoms = traj_rot_pt[j][i].copy()
#    
#            phi = atoms.get_dihedral(*dihedral_indices)
#            r = atoms.get_distance(0, 1)
#    
#            traj_grid_flat.append(atoms)
#    
#    write_xyz_traj("rot_then_liic_grid.xyz", traj_grid_flat)
#
#
#    G = get_G_matrix(qr=q_s, q_phi=q_phi, positions=positions_au, masses=masses, edge_order=2, method="fd", plot=True)
