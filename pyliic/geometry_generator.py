#! /usr/bin/env python3
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

from .interpolation import liic
from .utils import XYZ, compute_distance


def get_rdiff(atoms, proton_idx, atom1_idx, atom2_idx):
    pos = atoms.get_positions()

    H = pos[proton_idx]
    R1 = pos[atom1_idx]
    R2 = pos[atom2_idx]

    r1 = np.linalg.norm(H - R1)
    r2 = np.linalg.norm(H - R2)

    return r2 - r1


class GeometryGenerator:
    def __init__(
        self,
        react: XYZ,
        prod: XYZ,
        proton_idx: int,
        atom1_idx: int,
        atom2_idx: int | None = None,
        q1: str | None = "r1",
        q2: str | None = None,
        dihedral_indices=None,
        moving_indices=None,
        phi_min=None,
        phi_max=None,
        n_images: int = 100,
        order=None,
        int_method: str = "pchip",
        duplicate_tol: float = 1e-12):
        self.q1 = q1
        self.q2 = q2

        self.react = react
        self.prod = prod

        self.proton_idx = proton_idx
        self.atom1_idx = atom1_idx
        self.atom2_idx = atom2_idx

        self.dihedral_indices = dihedral_indices
        self.moving_indices = moving_indices
        self.phi_min = phi_min
        self.phi_max = phi_max

        self.symbols = react.get_symbols()

        if self.q1 is not None:
            self.mode = "q1"
        
            traj = liic(
                react,
                prod,
                n_images=n_images,
                order=order,
            )
        
            self.traj = traj
            self.positions = np.array(
                [g.get_positions() for g in traj],
                dtype=float,
            )
        
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
        
        elif self.q2 is not None:
            self.mode = "q2"
        
            if self.q2 != "dihedral":
                raise ValueError(f"Unknown q2='{self.q2}'. Expected 'dihedral'.")
        
            if self.phi_min is None or self.phi_max is None:
                raise ValueError("phi_min and phi_max cannot be None.")
        
            if self.dihedral_indices is None or self.moving_indices is None:
                raise ValueError(
                    "dihedral_indices and moving_indices are required for dihedral-only mode."
                )
        
            self.q = np.linspace(float(self.phi_min), float(self.phi_max), n_images)
        
            self.traj = []
            for phi in self.q:
                a = prod.copy()
                a.set_dihedral(
                    *self.dihedral_indices,
                    float(phi),
                    indices=self.moving_indices,
                )
                print(a.get_dihedral(*self.dihedral_indices))
                self.traj.append(a)
        
            self.positions = np.array(
                [g.get_positions() for g in self.traj],
                dtype=float,
            )
        
        else:
            raise NotImplementedError("Either q1 or q2 must be specified.") 

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
        elif self.q1 == "s":
            q = np.linspace(0.0, 1.0, len(self.positions))
        else:
            raise ValueError(f"Unknown q1='{self.q1}'. Expected 'r1', 'r2', or 'dr'.")

        if np.any(~np.isfinite(q)):
            raise ValueError("Non-finite q values found.")

        return q

    def __call__(self, q, phi=None, tol=1e-10):
        q_requested = float(q)
    
        qmin = self.qmin
        qmax = self.qmax
    
        if q_requested < qmin - tol or q_requested > qmax + tol:
            raise ValueError(
                f"Requested q={q_requested:.12f} outside interpolation range "
                f"[{qmin:.12f}, {qmax:.12f}]"
            )
    
        q_eval = np.clip(q_requested, qmin, qmax)
    
        pos = self.interpolator(q_eval)
    
        geom = XYZ(
            symbols=self.symbols,
            positions=np.asarray(pos, dtype=float),
        )
    
        if self.mode == "q1" and phi is not None:
            if self.dihedral_indices is None or self.moving_indices is None:
                raise ValueError(
                    "dihedral_indices and moving_indices are required "
                    "when calling the generator with phi."
                )
    
            geom.set_dihedral(
                *self.dihedral_indices,
                float(phi),
                indices=self.moving_indices,
            )
    
        # In q2-only mode, phi argument should not be used
        if self.mode == "q2" and phi is not None:
            raise ValueError(
                "This generator is already in dihedral-only mode. "
                "Call it as gen(phi), not gen(q, phi=...)."
            )
    
        return geom
    
    @property
    def qmin(self):
        return float(np.min(self.q))
    
    @property
    def qmax(self):
        return float(np.max(self.q))
    
    @property
    def q1min(self):
        return self.qmin
    
    @property
    def q1max(self):
        return self.qmax
