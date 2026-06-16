#! /usr/bin/env python3

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

from .interpolation import liic
from .utils import XYZ, compute_distance
from .eckart import apply_eckart


def get_rdiff(atoms, proton_idx, atom1_idx, atom2_idx):
    pos = atoms.get_positions()

    H = pos[proton_idx]
    R1 = pos[atom1_idx]
    R2 = pos[atom2_idx]

    r1 = np.linalg.norm(H - R1)
    r2 = np.linalg.norm(H - R2)

    return r2 - r1


class LIICGeometryGenerator:
    def __init__(
        self,
        react: XYZ,
        prod: XYZ,
        proton_idx: int,
        atom1_idx: int,
        atom2_idx: int | None = None,
        q1: str = "r1",
        q2: str | None = None,
        dihedral_indices=None,
        moving_indices=None,
        n_images: int = 100,
        order=None,
        int_method: str = "pchip",
        duplicate_tol: float = 1e-12,
    ):
        self.q1 = q1
        self.q2 = q2

        self.react = react
        self.prod = prod

        self.proton_idx = proton_idx
        self.atom1_idx = atom1_idx
        self.atom2_idx = atom2_idx

        self.dihedral_indices = dihedral_indices
        self.moving_indices = moving_indices

        self.symbols = react.get_symbols()

        traj = liic(
            react,
            prod,
            n_images=n_images,
            order=order)

        self.traj = traj
        self.positions = np.array(
            [g.get_positions() for g in traj],
            dtype=float)

        self.q = self._compute_q()

        sort_idx = np.argsort(self.q)
        self.q = self.q[sort_idx]
        self.positions = self.positions[sort_idx]

        dq = np.diff(self.q)

        if np.any(np.abs(dq) < duplicate_tol):
            raise ValueError(
                f"Duplicate or nearly duplicate q values found for q1='{q1}'. "
                f"Minimum |dq| = {np.min(np.abs(dq)):.6e}. "
                "The mapping q -> geometry is not unique."
            )

        if not np.all(dq > 0):
            raise ValueError(
                "q values are not strictly increasing after sorting. "
                "This should not happen unless there are NaNs or duplicate values."
            )

        int_method = int_method.lower()

        if int_method == "pchip":
            self.interpolator = PchipInterpolator(
                self.q,
                self.positions,
                axis=0,
                extrapolate=False,
            )
        elif int_method == "cubic":
            self.interpolator = CubicSpline(
                self.q,
                self.positions,
                axis=0,
                extrapolate=False,
            )
        else:
            raise NotImplementedError(
                f"Unknown interpolation method: {int_method}"
            )

    def _compute_q(self):
        if self.q1 == "r1":
            q = compute_distance(
                self.positions[:, self.proton_idx, :],
                self.positions[:, self.atom1_idx, :],
                axis=-1)

        elif self.q1 == "r2":
            if self.atom2_idx is None:
                raise ValueError("atom2_idx is required when q1='r2'.")

            q = compute_distance(
                self.positions[:, self.proton_idx, :],
                self.positions[:, self.atom2_idx, :],
                axis=-1)

        elif self.q1 == "dr":
            if self.atom2_idx is None:
                raise ValueError("atom2_idx is required when q1='dr'.")

            q = np.array([
                get_rdiff(g, self.proton_idx, self.atom1_idx, self.atom2_idx) for g in self.traj])

        else:
            raise ValueError(f"Unknown q1='{self.q1}'. Expected 'r1', 'r2', or 'dr'.")

        if np.any(~np.isfinite(q)):
            raise ValueError("Non-finite q values found.")

        return q

    def __call__(self, q, phi=None, enforce=False, tol=1e-10):
        q_requested = float(q)
    
        qmin = self.q1min
        qmax = self.q1max
    
        # allow tiny floating-point overshoot at the boundaries
        if q_requested < qmin - tol or q_requested > qmax + tol:
            raise ValueError(
                f"Requested q={q_requested:.12f} outside interpolation range "
                f"[{qmin:.12f}, {qmax:.12f}]"
            )
    
        # clamp if very close to boundary
        q_eval = np.clip(q_requested, qmin, qmax)
    
        pos = self.interpolator(q_eval)
    
        geom = XYZ(
            symbols=self.symbols,
            positions=np.asarray(pos, dtype=float),
        )
    
        if phi is not None:
            if self.dihedral_indices is None or self.moving_indices is None:
                raise ValueError("dihedral_indices and moving_indices are required when calling the generator with phi")
            geom.set_dihedral(
                *self.dihedral_indices,
                float(phi),
                indices=self.moving_indices,
            )
    
#        if enforce:
#            if self.q1 == "dr":
#                geom = enforce_rdiff_keep_perp(
#                    geom,
#                    q=q_requested,
#                    proton_idx=self.proton_idx,
#                    atom1_idx=self.atom1_idx,
#                    atom2_idx=self.atom2_idx,
#                )
#    
#            elif self.q1 == "r1":
#                geom = enforce_distance_to_atom(
#                    geom,
#                    r=q_requested,
#                    proton_idx=self.proton_idx,
#                    atom_idx=self.atom1_idx,
#                )
#    
#            elif self.q1 == "r2":
#                geom = enforce_distance_to_atom(
#                    geom,
#                    r=q_requested,
#                    proton_idx=self.proton_idx,
#                    atom_idx=self.atom2_idx,
#                )
        return geom
    
    @property
    def q1min(self):
        return float(np.min(self.q))

    @property
    def q1max(self):
        return float(np.max(self.q))
