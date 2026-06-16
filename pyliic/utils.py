#! /usr/bin/env python3
from typing import DefaultDict
import numpy as np

from .data import AMU2AU, chemical_symbols, atomic_numbers, atomic_masses_common, atomic_masses_iupac2016


def normalize(v, tol=1e-12):
    norm = np.linalg.norm(v)
    if norm < tol:
        raise ValueError()
    return v / norm


def place_atom(p0, p1, p2, r, theta_deg, phi_deg):
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    
    p0 = np.array(p0)
    p1 = np.array(p1)
    p2 = np.array(p2)
    # unit vector from p0 to p1
    e1 = normalize(p0 - p1)
    # vector normal to plane p0, p1, p2
    n = np.cross(p0 - p1, p2 - p1)
    e3 = normalize(n)
    # third orthogonal vector in local frame
    e2 = np.cross(e3, e1)
    # local displacement from p0 to p1 (from local spherical to local cartesian)
    d_local = -r * np.cos(theta) * e1 + r * np.sin(theta) * np.cos(phi) * e2 + r * np.sin(theta) * np.sin(phi) * e3
    return p0 + d_local


def compute_distance(p0, p1):
    return np.linalg.norm(p1 - p0)


def compute_angle( p0, p1, p2):
    d1 = p0 - p1
    d2 = p2 - p1
    norm1 = np.linalg.norm(d1)
    norm2 = np.linalg.norm(d2)

    cos_theta = np.dot(d2, d1) / (norm2 * norm1)
    theta = np.arccos(cos_theta)
    return np.rad2deg(theta)


def compute_dihedral(p0, p1, p2, p3):
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    b1 /= np.linalg.norm(b1)

    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    dihedral = np.atan2(y, x)
    if dihedral < 0:
        dihedral += 2 * np.pi
    return np.rad2deg(dihedral)


def rotate(center, rotation_axis, coordinates, theta, mask):
    theta = np.radians(theta)

    coords = coordinates[mask]

    coords -= center
    k = rotation_axis / np.linalg.norm(rotation_axis)
    s = np.sin(theta)
    c = np.cos(theta)
    coords += np.cross(s * k, coords) + np.cross((1 - c) * k, np.cross(k, coords))
    coords += center

    coordinates[mask] = coords
    return coordinates



class ZMatrix():
    def __init__(self, symbols,
                 bonds, bond_refs,
                 angles, angle_refs,
                 dihedrals, dihedral_refs):
        self.symbols = symbols
        self.bonds = bonds
        self.angles = angles
        self.dihedrals = dihedrals
        self.dihedral_refs = dihedral_refs
        self.bond_refs = bond_refs
        self.angle_refs = angle_refs
        self.n_atoms = len(symbols)
    
    def __repr__(self) -> str:
        return f"ZMatrix({self.symbols=}, {self.bond_refs=}, {self.bonds}, {self.angle_refs=}, {self.angles}, {self.dihedral_refs}, {self.dihedrals})"

    def __str__(self):
        lines = [f"ZMatrix object with {self.n_atoms} atoms"]
    
        for i in range(self.n_atoms):
            symbol = self.symbols[i]
    
            if i == 0:
                line = f"{i:3d}: {symbol:3s}"
    
            elif i == 1:
                line = (
                    f"{i:3d}: {symbol:3s} "
                    f"{self.bond_refs[i]:3d} "
                    f"{self.bonds[i]:12.6f}")
    
            elif i == 2:
                line = (
                    f"{i:3d}: {symbol:3s} "
                    f"{self.bond_refs[i]:3d} "
                    f"{self.bonds[i]:12.6f} "
                    f"{self.angle_refs[i]:3d} "
                    f"{self.angles[i]:12.6f}")
    
            else:
                line = (
                    f"{i:3d}: {symbol:3s} "
                    f"{self.bond_refs[i]:3d} "
                    f"{self.bonds[i]:12.6f} "
                    f"{self.angle_refs[i]:3d} "
                    f"{self.angles[i]:12.6f} "
                    f"{self.dihedral_refs[i]:3d} "
                    f"{self.dihedrals[i]:12.6f}")
    
            lines.append(line)
    
        return "\n".join(lines)

    def to_xyz(self):
        positions = np.zeros((self.n_atoms, 3))

        positions[0] = np.array([0., 0., 0.])
        if self.n_atoms == 1:
            return XYZ(self.symbols, positions)

        j = self.bond_refs[1]
        r = self.bonds[1]

        positions[1] = positions[j] + np.array([r, 0., 0.])

        if self.n_atoms == 2:
            return XYZ(self.symbols, positions)

        j = self.bond_refs[2]
        k = self.angle_refs[2]
        r = self.bonds[2]
        theta = np.deg2rad(self.angles[2])

        a = positions[j]
        b = positions[k]

        e1 = normalize(a - b)

        tmp_axis = np.array([0., 0., 1.])
        if abs(np.dot(e1, tmp_axis)) > 0.99:
            tmp_axis = np.array([0., 1., 0.])
        e2 = normalize(np.cross(tmp_axis, e1))

        positions[2] = a - r * np.cos(theta) * e1 + r * np.sin(theta) * e2

        for i in range(3, self.n_atoms):
            j = self.bond_refs[i]
            k = self.angle_refs[i]
            l = self.dihedral_refs[i]
            
            r = self.bonds[i]
            theta = self.angles[i]
            phi = self.dihedrals[i]

            positions[i] = place_atom(positions[j], positions[k], positions[l], r, theta, phi)
        
        return XYZ(self.symbols, positions)
   
    @classmethod
    def read_zmat(cls, filename: str):
        symbols, bonds, bond_refs, angles, angle_refs, dihedrals, dihedral_refs = [], [], [], [], [], [], []
        with open(filename, "r") as f:
            for line in f:
                line = line.strip().split()
                print(line)
                if len(line) == 1:
                    symbols.append(line[0])
                    bonds.append(None)
                    bond_refs.append(None)
                    angles.append(None)
                    angle_refs.append(None)
                    dihedrals.append(None)
                    dihedral_refs.append(None)
                elif len(line) == 3:
                    symbols.append(line[0])
                    bonds.append(float(line[2]))
                    bond_refs.append(int(line[1]))
                    angles.append(None)
                    angle_refs.append(None)
                    dihedrals.append(None)
                    dihedral_refs.append(None)
                elif len(line) == 5:
                    symbols.append(line[0])
                    bonds.append(float(line[2]))
                    bond_refs.append(int(line[1]))
                    angles.append(float(line[4]))
                    angle_refs.append(int(line[3]))
                    dihedrals.append(None)
                    dihedral_refs.append(None)
                elif len(line) > 5:
                    symbols.append(line[0])
                    bonds.append(float(line[2]))
                    bond_refs.append(int(line[1]))
                    angles.append(float(line[4]))
                    angle_refs.append(int(line[3]))
                    dihedrals.append(float(line[6]))
                    dihedral_refs.append(int(line[5]))
                else:
                    raise ValueError('Wrong ZMAT file!')

        return cls(symbols, bonds, bond_refs, angles, angle_refs, dihedrals, dihedral_refs)

    def copy(self):
        return ZMatrix(
            symbols=self.symbols.copy(),
            bonds=self.bonds.copy(),
            bond_refs=self.bond_refs.copy(),
            angles=self.angles.copy(),
            angle_refs=self.angle_refs.copy(),
            dihedrals=self.dihedrals.copy(),
            dihedral_refs=self.dihedral_refs.copy())

    def write_zmat(self, filename: str):
        with open(filename, "w") as f:
            for i in range(self.n_atoms):
                symbol = self.symbols[i]
                if i == 0:
                    line = f"{symbol:3s}"#{0:3d} {0.:12.10f} {0:3d} {0.:12.10f} {0:3d} {0.:12.10f}"
                elif i == 1:
                    line = f"{symbol:3s} {self.bond_refs[i] + 1:3d} {self.bonds[i]:12.10f}" #+ \
#                            f" {0:3d} {0.:12.10f} {0:3d} {0.:12.10f}"
                elif i == 2:
                    line = f"{symbol:3s} {self.bond_refs[i] + 1:3d} {self.bonds[i]:12.10f}" + \
                            f" {self.angle_refs[i] + 1:3d} {self.angles[i]:12.10f}"# + \
#                            f" {0:3d} {0.:12.10f}"
                else:
                    line = f"{symbol:3s} {self.bond_refs[i] + 1:3d} {self.bonds[i]:12.10f}" + \
                            f" {self.angle_refs[i] + 1:3d} {self.angles[i]: 12.10f}" + \
                            f" {self.dihedral_refs[i] + 1:3d} {self.dihedrals[i]:12.10f}"

                f.write(f"{line}\n")


class XYZ():
    def __init__(self, symbols, positions):
        if len(symbols) != len(positions):
            raise ValueError()
        if positions.ndim != 2:
            raise ValueError("Positions array should be 2D (n_atoms, 3)!")

        self.symbols = symbols
        self.positions = np.asarray(positions)
        self.n_atoms = len(positions)

    def __repr__(self):
        values = self._count_symbols()
        values.sort(key=lambda s: s[0])
        symbols = []
        for v in values:
            v[-1] = str(v[-1])
            symbols.append(''.join(v))
        
        return f"XYZ(n_atoms={self.n_atoms}, symbols={''.join(symbols)})"

    def __str__(self):
        lines = [f"XYZ object with {self.n_atoms} atoms"]
    
        for symbol, position in zip(self.symbols, self.positions):
            lines.append(
                f"{symbol:>3} "
                f"{position[0]:12.6f} "
                f"{position[1]:12.6f} "
                f"{position[2]:12.6f}"
            )
    
        return "\n".join(lines)

    def _count_symbols(self):
        values = [[x, self.symbols.count(x)] for x in set(self.symbols)]
        return values
    
    def set_positions(self, new_positions):
        self.positions = new_positions

    def get_masses_amu(self):
        numbers = np.array([atomic_numbers.get(symbol) for symbol in self.symbols])
        masses = atomic_masses_common[numbers]
        return masses

    def get_masses_au(self):
        numbers = np.array([atomic_numbers.get(symbol) for symbol in self.symbols])
        masses = atomic_masses_common[numbers] * AMU2AU
        return masses

    def get_distance(self, a0, a1):
        return compute_distance(self.positions[a0], self.positions[a1])

    def get_angle(self, a0, a1, a2):
        return compute_angle(self.positions[a0], self.positions[a1], self.positions[a2])

    def get_dihedral(self, a0, a1, a2, a3):
        return compute_dihedral(self.positions[a0], self.positions[a1], self.positions[a2], self.positions[a3])

    def get_positions(self):
        xyz_cpy = self.copy()
        return np.array(xyz_cpy.positions)

    def set_dihedral(self, a0: int, a1: int, a2: int, a3: int,
                     theta: float,
                     indices: list | np.ndarray | tuple = None) -> np.ndarray:
        """
        Adjust dihedral angle between two planes defined by atoms with indices
        a0, a1, a2, and a3.
        Indices is an array consisting of other atioms that are changed during
        the rotation.
        """
        rotated_coordinates = np.zeros((self.positions.shape))
    
        coordinates = np.asarray(self.positions)
    
        curr_angle = compute_dihedral(self.positions[a0], self.positions[a1], self.positions[a2], self.positions[a3])
        diff = theta - curr_angle
    
        if indices is None:
            mask = np.zeros(len(coordinates))
            mask[a3] = 1
        else:
            mask = [index in indices for index in range(len(self.positions))]
    
        # rotate aroud an axis by the angle difference
        center = self.positions[a2]
        rotation_axis = self.positions[a2] - self.positions[a1]
        coordinates = rotate(center, rotation_axis, self.positions, diff, mask)
    
        new_angle = compute_dihedral(self.positions[a0], self.positions[a1], self.positions[a2], self.positions[a3])
        return coordinates


    def to_zmat(self, ref_atom_idx=0):
        ref_symbol = self.symbols[ref_atom_idx]
        ref_positions = self.positions[ref_atom_idx]
        n_atoms = self.n_atoms
        symbols = [ref_symbol]
        positions = [ref_positions]
        original_indices = [ref_atom_idx]
        for i, (symbol, position) in enumerate(zip(self.symbols, self.positions)):
            if i == ref_atom_idx:
                continue
            symbols.append(symbol)
            positions.append(position)
            original_indices.append(i)
    
        positions = np.asarray(positions)

        bonds = [None] * n_atoms
        bond_refs = [None] * n_atoms
       
        angles = [None] * n_atoms
        angle_refs = [None] * n_atoms

        dihedrals = [None] * n_atoms
        dihedral_refs = [None] * n_atoms
        
        for i in range(n_atoms):
            if i == 0:
                continue
            elif i == 1:
                bond_refs[i] = 0

                bonds[i] = compute_distance(positions[i], positions[0])
            elif i == 2:
                bond_refs[i] = 0
                angle_refs[i] = 1

                bonds[i] = compute_distance(positions[i], positions[bond_refs[i]])
                angles[i] = compute_angle(positions[i], positions[bond_refs[i]], positions[angle_refs[i]])
            else:
                bond_refs[i] = i - 1
                angle_refs[i] = i - 2
                dihedral_refs[i] = i - 3

                bonds[i] = compute_distance(positions[i], positions[bond_refs[i]])
                angles[i] = compute_angle(positions[i], positions[bond_refs[i]], positions[angle_refs[i]])
                dihedrals[i] = compute_dihedral(positions[i], positions[bond_refs[i]], positions[angle_refs[i]], positions[dihedral_refs[i]])
        
        return ZMatrix(symbols=symbols,
                       bonds=bonds,
                       bond_refs=bond_refs,
                       angles=angles,
                       angle_refs=angle_refs,
                       dihedrals=dihedrals,
                       dihedral_refs=dihedral_refs)

    def reorder_atoms(self, order):
        order = list(order)
        if len(order) != self.n_atoms:
            raise ValueError()
        
        symbols = [self.symbols[i] for i in order]
        positions = self.positions[order]

        return XYZ(symbols, positions)


    def write_xyz(self, filename: str) -> None:
        with open(filename, "w") as f:
            f.write(f"{self.n_atoms}\n\n")
            for symbol, position in zip(self.symbols, self.positions):
                line = f"{symbol: >3} {position[0]:10.10f} {position[1]:10.10f} {position[2]:10.10f}\n"
                f.write(line)

    def copy(self):
        return XYZ(symbols=self.symbols.copy(), positions=self.positions.copy())

    @classmethod
    def read_xyz(cls, filename: str):
        geometries = []
        with open(filename, "r") as f:
            while True:
                line = f.readline()
                
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    n_atoms = int(line)
                except ValueError:
                    raise ValueError(f"Expected number of atoms, but {line}")
                f.readline()

                symbols, positions = [], []
                for _ in range(n_atoms):
                    line = f.readline()
                    if not line:
                        raise ValueError(f"Unexpected end of file.")
                    parts = line.split()
                    if len(parts) < 4:
                        raise ValueError(f"Invalid atom line: {line.strip()}")
                    symbols.append(parts[0])
                    positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
                geometries.append(cls(symbols, np.asarray(positions)))
        if len(geometries) == 1:
            return geometries[0]
        return geometries


if __name__ == "__main__":
    atoms_lst = XYZ.read_xyz("sac_hdisplaced_bounded.xyz")
    atoms = atoms_lst[0]
    print(atoms)
    atoms = atoms.reorder_atoms([9, 11, 8, 0, 1, 2, 3, 4, 5, 6, 7, 10, 12, 13, 14, 15])
    print(atoms)
    zmat = atoms.to_zmat()
    print(zmat)
    atoms0 = zmat.to_xyz()
    atoms0.write_xyz("test_sac.xyz")
