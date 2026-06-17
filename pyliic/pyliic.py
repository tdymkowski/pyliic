#! /usr/bin/env python3
import numpy as np
from pathlib import Path
import sys
from os.path import join
from pyliic.eckart import apply_eckart
from .fileio import write_xyz, write_xyz_traj
from .gmatrix import get_G_element, get_G_matrix, reshape_positions_for_gmat
from .data import AMU2AU, ANG2AU
from .interpolation import liic, make_distance_dihedral_grid
from .utils import XYZ
from .geometry_generator import GeometryGenerator

N_IMAGES = 200
GEOM_DATA_FILE = "geometry_data.txt"

def read_pyphspu_input():
    xyzpath = sys.argv[1]
    tab = np.genfromtxt(sys.argv[2])
    if tab.ndim > 1:
        idx = tab[:, 0]
        x = tab[:, 1]
    else:
        idx = np.array(tab[0])
        x = np.array(tab[1])
    outpath = sys.argv[3]
    return xyzpath, outpath, idx, x


def create_geometries_ro1h(n_images=200):
    xyzpath, outpath, _, RO1H = read_pyphspu_input()
    xyz_lst = XYZ.read_xyz(xyzpath)
    react = xyz_lst[0]
    prod = xyz_lst[1]
    gen_licc = GeometryGenerator(
        react=react,
        prod=prod,
        proton_idx=11,
        atom1_idx=9,
        atom2_idx=8,
        q1="r1",
        q2=None,
        n_images=n_images,
        int_method="pchip")
    data = []
    geoms = []
    for r in RO1H:
        geom_r = gen_licc(r)
        geoms.append(geom_r)
    geoms = apply_eckart(geoms)
    for i, (geom, r) in enumerate(zip(geoms, RO1H)):
        fname = f"sac_{i}.xyz"
        path = join(outpath, fname)
        write_xyz(path, geom)
        data.append([path, r])
    return data


def create_geometries_ro2h(n_images=200):
    xyzpath, outpath, _, RO2H = read_pyphspu_input()
    xyz_lst = XYZ.read_xyz(xyzpath)
    react = xyz_lst[0]
    prod = xyz_lst[1]
    gen_licc = GeometryGenerator(
        react=react,
        prod=prod,
        proton_idx=11,
        atom1_idx=9,
        atom2_idx=8,
        q1="r2",
        q2=None,
        n_images=n_images,
        int_method="pchip")
    data = []
    geoms = []
    for r in RO2H:
        geom_r = gen_licc(r)
        geoms.append(geom_r)

    geoms = apply_eckart(geoms)

    for i, (geom, r) in enumerate(zip(geoms, RO2H)):
        fname = f"sac_{i}.xyz"
        path = join(outpath, fname)
        write_xyz(path, geom)
        data.append([path, r])
    return data


def create_geometries_dr(n_images=200):
    # r2 - r1
    xyzpath, outpath, _, DR = read_pyphspu_input()
    xyz_lst = XYZ.read_xyz(xyzpath)
    react = xyz_lst[0]
    prod = xyz_lst[1]
    gen_licc = GeometryGenerator(
        react=react,
        prod=prod,
        proton_idx=11,
        atom1_idx=9,
        atom2_idx=8,
        q1="dr",
        q2=None,
        n_images=n_images,
        int_method="pchip")
    data = []
    geoms = []
    for dr in DR:
        geom_r = gen_licc(dr)
        geoms.append(geom_r)

    geoms = apply_eckart(geoms)

    for i, (geom, dr) in enumerate(zip(geoms, DR)):
        fname = f"sac_{i}.xyz"
        path = join(outpath, fname)
        write_xyz(path, geom)
        data.append([path, dr])
    return data


def create_geometries_phi(n_images=200):
    # dihedral
    xyzpath, outpath, _, PHI = read_pyphspu_input()
    xyz_lst = XYZ.read_xyz(xyzpath)
    react = xyz_lst[0]
    prod = xyz_lst[1]
    gen_phi = GeometryGenerator(
        react=react,
        prod=prod,
        proton_idx=11,
        atom1_idx=9,
        q1=None,
        q2="dihedral",
        dihedral_indices=[1, 0, 6, 8],
        moving_indices=[8, 7, 10, 11],
        phi_min=0.0,
        phi_max=360.0,
        n_images=n_images,
        int_method="pchip")
    data = []
    geoms = []

    for i, phi in enumerate(PHI):
        geom_r = gen_phi(phi)
        geoms.append(geom_r)

    geoms = apply_eckart(geoms)
    for i, (geom, phi) in zip(geoms, PHI):
        fname = f"sac_{i}.xyz"
        path = join(outpath, fname)
        write_xyz(path, geom)
        data.append([path, phi])
    return data


def create_geometries_2D(q1="r1", n_images=200):
    xyzpath, outpath, _, X = read_pyphspu_input()
    xyz_lst = XYZ.read_xyz(xyzpath)
    react = xyz_lst[0]
    prod = xyz_lst[1]
    make_geom = GeometryGenerator(react,
                                  prod,
                                  proton_idx=11,
                                  atom1_idx=9,
                                  atom2_idx=8,
                                  dihedral_indices=[1, 0, 6, 8],
                                  moving_indices=[8, 7, 10, 11],
                                  q1=q1,
                                  n_images=n_images)
    data = []
    geoms = []
    for r, phi in X:
        geom_r_phi = make_geom(r, phi=phi)
        geoms.append(geom_r_phi)
    geoms = apply_eckart(geoms)
    for i, (geom, (r, phi)) in enumerate(zip(geoms, X[:, 0])):
        fname = f"sac_{i}.xyz"
        path = join(outpath, fname)
        write_xyz(path, geom)
        data.append([path, [r, phi]])
    return data


def write_pyphspu_data(data, dirpath, filename="geometry_data.txt"):
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)

    outfile = dirpath / filename

    with open(outfile, "w") as f:
        for item in data:
            if len(item) < 2:
                raise ValueError(
                    "Each data entry must contain at least outpath and one coordinate."
                )

            outpath = item[0]

            if len(item) == 2:
                coords = np.asarray(item[1], dtype=float).ravel()
            else:
                coords = np.asarray(item[1:], dtype=float).ravel()

            if coords.size < 1 or coords.size > 4:
                raise ValueError(
                    f"Coordinates must have dimension 1–4. "
                    f"Got dimension {coords.size} for {outpath}."
                )

            coord_str = " ".join(f"{x:>10.6f}" for x in coords)

            f.write(f"{str(outpath):>40s} {coord_str}\n")

    return outfile
        

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

    make_geom = GeometryGenerator(react, prod, proton_idx=11, atom1_idx=9, atom2_idx=8, dihedral_indices=[1, 0, 6, 8], moving_indices=[8, 7, 10, 11], q1="s")

    r1_fine = np.linspace(np.min(q1_r), np.max(q1_r), 200)
    r2_fine = np.linspace(np.max(q2_r), np.min(q2_r), 200)

    q1_fine = np.linspace(0, 1, 200)
    q2_fine = np.linspace(0., 360., 200)
    geoms = []
    for q1_f in q1_fine:
        for q2_f in q2_fine:
            geom_q1 = make_geom(q1_f, phi=q2_f)
            geoms.append(geom_q1)
    geoms = apply_eckart(geoms)
    write_xyz_traj("traj_liic.xyz", geoms)

    q1_label = r"$s$"
    positions = np.array([g.get_positions() for g in geoms]) * ANG2AU
    positions = reshape_positions_for_gmat(q1_fine, q2_fine, positions)
    n_q1 = len(q1_fine)
    n_q2 = len(q2_fine)
    n_atoms = positions.shape[1]
#    positions = positions.reshape(n_q1, n_q2, n_atoms, 3)
#    q1_fine *= ANG2AU
    q2_fine_rad = np.deg2rad(q2_fine)
    G = get_G_matrix(q1_fine, q2_fine_rad, positions, masses, plot=True, save_op=False, method="fd", q1_label=q1_label)



    gen_phi = GeometryGenerator(
        react=react,
        prod=prod,
        proton_idx=11,
        atom1_idx=9,
        q1=None,
        q2="dihedral",
        dihedral_indices=[1, 0, 6, 8],
        moving_indices=[8, 7, 10, 11],
        phi_min=0.0,
        phi_max=360.0,
        n_images=200,
        int_method="pchip",
    )
    
    q2_fine = np.linspace(0., 360., 400)
    geoms2 = []
    for q2_f in q2_fine:
        geom_q1 = gen_phi(q2_f)
        geoms2.append(geom_q1)
 
    geoms2 = apply_eckart(geoms2)
    positions = np.array([g.get_positions() for g in geoms2]) * AMU2AU
    G = get_G_element(q2_fine, positions, masses, plot=True)
    write_xyz_traj("traj_rot.xyz", geoms2)

    gen_licc = GeometryGenerator(
        react=react,
        prod=prod,
        proton_idx=11,
        atom1_idx=9,
        atom2_idx=8,
        q1="s",
        q2=None,
#        dihedral_indices=[1, 0, 6, 8],
#        moving_indices=[8, 7, 10, 11],
#        phi_min=0.0,
#        phi_max=360.0,
        n_images=200,
        int_method="pchip",
    )
    q1_r = np.array([atoms.get_distance(0, 1) for atoms in traj_pt])
    q2_r = np.array([atoms.get_distance(1, 2) for atoms in traj_pt])

    r1_fine = np.linspace(np.min(q1_r), np.max(q1_r), 400)
    r2_fine = np.linspace(np.max(q2_r), np.min(q2_r), 400)
    q1_fine = r2_fine - r1_fine
    geoms = []
    q1_fine = np.linspace(0, 1, 400)
    for q1_f in q1_fine:

        geom_q1 = gen_licc(q1_f)
        geoms.append(geom_q1)
 
    geoms = apply_eckart(geoms)
    positions = np.array([g.get_positions() for g in geoms]) * AMU2AU
#    G = get_G_element(q1_fine, positions, masses, plot=True)
    write_xyz_traj("traj_liic_r1.xyz", geoms)


#    q1_label = r"$r{O1H}$"
#    positions = np.array([g.get_positions() for g in geoms]) * ANG2AU
#    n_q1 = len(q1_fine)
#    n_q2 = len(q2_fine)
#    n_atoms = positions.shape[1]
#    positions = positions.reshape(n_q1, n_q2, n_atoms, 3)
#    q1_fine *= ANG2AU
#    q2_fine_rad = np.deg2rad(q2_fine)
#    G = get_G_matrix(q1_fine, q2_fine_rad, positions, masses, plot=True, save_op=False, method="fd", q1_label=q1_label)
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
