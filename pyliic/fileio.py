#! /usr/bin/env python3
from .utils import XYZ
from numpy import ndarray

def write_xyz_traj(filename: str, xyz_traj: list | ndarray):
    with open(filename, "w") as f:
        for xyz in xyz_traj:
            symbols = xyz.symbols
            positions = xyz.positions
            n_atoms = xyz.n_atoms
            f.write(f"{n_atoms}\n\n")
            for symbol, position in zip(symbols, positions):
                line = f"{symbol: >3} {position[0]:10.10f} {position[1]:10.10f} {position[2]:10.10f}\n"
                f.write(line)

def write_xyz(filename: str, xyz: XYZ):
    symbols = xyz.symbols
    positions = xyz.positions
    n_atoms = xyz.n_atoms
    with open(filename, "w") as f:
        f.write(f"{n_atoms}\n\n")
        for symbol, position in zip(symbols, positions):
            line = f"{symbol: >3} {position[0]:10.10f} {position[1]:10.10f} {position[2]:10.10f}\n"
            f.write(line)
