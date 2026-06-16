#! /usr/bin/env python3
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
import matplotlib.pyplot as plt


def get_dxdq_fd(q, positions, edge_order=2, axis=0):
    order = np.argsort(q)
    q = q[order]
    if any(np.diff(q) <= 0.):
        raise ValueError(f"Internal coordinate must be striclty increasing sequence!")
    x = positions[order]
    
    dxdq = np.gradient(x, q, axis=axis, edge_order=edge_order)
    return dxdq


def get_dxdq_spline(q, positions, spline_type="cubic", axis=0, spline_kwargs={}):
    spline_methods = {"cubic": CubicSpline, "pchip": PchipInterpolator}
    order = np.argsort(q)
    q = q[order]
    if any(np.diff(q) <= 0.):
        raise ValueError(f"Internal coordinate must be striclty increasing sequence!")
    x = positions[order]
    spline_method = spline_methods[spline_type]
    spline = spline_method(q, x, axis=axis, **spline_kwargs)
    dxdq = spline.derivative(1)(q)
    return dxdq


def calc_inv_G_value(dxdq: np.ndarray, masses: np.ndarray):
    invG = np.sum(masses[None, :, None] * dxdq**2, axis=(1, 2))
    return invG


def get_inv_G_value(q, positions, masses, der_type=""):
    positions = np.asarray(positions)
    if der_type.lower() == "fd":
        dxdq = get_dxdq_fd(q, positions)
    elif der_type.lower() == "pchip":
        dxdq = get_dxdq_spline(q, positions, spline_type="pchip")
    else:
        dxdq = get_dxdq_spline(q, positions, spline_type="cubic")

    invG_value = calc_inv_G_value(dxdq, masses)
    return invG_value

def get_dxdq(q, positions, method="cubic", axis=0, edge_order=2):
    if method.lower() == "fd":
        dxdq = get_dxdq_fd(q, positions, edge_order=edge_order, axis=axis)
    elif method.lower() == "pchip":
        dxdq = get_dxdq_spline(q, positions, spline_type="pchip", axis=axis)
    else:
        dxdq = get_dxdq_spline(q, positions, spline_type="cubic", axis=axis)
    return dxdq


def plot_G_matrix(q1, q2, G, invG=None, **kwargs):
    components = [
        (0, 0, r"$G_{rr}$"),
        (0, 1, r"$G_{rs}$"),
        (1, 0, r"$G_{sr}$"),
        (1, 1, r"$G_{ss}$"),
    ]

    R, P = np.meshgrid(q1, q2, indexing="ij")
    q1_label = kwargs.get("q1_label", r"$r_{O2H} - r_{O1H}$ (au)")
    q2_label = kwargs.get("q2_label", r"$\phi$ (rad)")
    levels = kwargs.get("levels", 30)
    cmap = kwargs.get("cmap", None)
    save = kwargs.get("save", False)
    filename = kwargs.get("filename", "gmat.pdf")

    fig, axes = plt.subplots(2, 2, figsize=kwargs.get("figsize", (10, 8)))

    for ax, (a, b, title) in zip(axes.flatten(), components):
        contour = ax.contourf(R, P, G[:, :, a, b], levels=levels, cmap=cmap)

        ax.set_xlabel(q1_label)
        ax.set_ylabel(q2_label)
        ax.set_title(title)

        fig.colorbar(contour, ax=ax, label=title)

    fig.tight_layout()

    if save:
        fig.savefig(filename, bbox_inches="tight")

    plt.show()


    if invG is not None:
        components = [
            (0, 0, "invG_rr"),
            (0, 1, "invG_rphi"),
            (1, 0, "invG_phir"),
            (1, 1, "invG_phiphi"),
        ]

        for a, b, title in components:
            plt.figure()
            plt.contourf(R, P, invG[:, :, a, b], levels=30)
            plt.xlabel("r / bohr")
            plt.ylabel("phi / rad")
            plt.colorbar(label=title)
            plt.title(title)
            plt.tight_layout()
            plt.show()


def get_G_matrix(qr, q_phi, positions, masses, edge_order=2, method="cubic", plot=False, **kwargs):
    n_r, n_phi, n_atoms, ndim = positions.shape

    dx_dr = get_dxdq(qr, positions, method=method, axis=0)
    dx_dphi = get_dxdq(q_phi, positions, method=method, axis=1)
    invG = np.zeros((n_r, n_phi, 2, 2))

    invG[:, :, 0, 0] = np.sum(masses[None, None, :, None] * dx_dr * dx_dr, axis=(2, 3))
    invG[:, :, 1, 1] = np.sum(masses[None, None, :, None] * dx_dphi * dx_dphi, axis=(2, 3))
    invG[:, :, 0, 1] = invG[:, :, 1, 0] = np.sum(masses[None, None, :, None] * dx_dr * dx_dphi, axis=(2, 3))

    G = np.linalg.inv(invG)

    if plot:
        plot_G_matrix(qr, q_phi, G, **kwargs)

    return G
