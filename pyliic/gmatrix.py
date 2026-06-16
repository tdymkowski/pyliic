#! /usr/bin/env python3
import numpy as np
import qdio.qd_file_op as qdop
from scipy.interpolate import CubicSpline, PchipInterpolator
import matplotlib.pyplot as plt


def reshape_positions_for_gmat(q1, q2, positions):
    positions0 = positions.copy()
    n_q1 = len(q1)
    n_q2 = len(q2)
    n_atoms = positions.shape[1]
    positions0 = positions0.reshape(n_q1, n_q2, n_atoms, 3)
    return positions0


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


def get_G_element(q, positions, masses, method="fd", axis=0, edge_order=2, plot=False, save=False):
    dxdq = get_dxdq(q, positions, method=method, axis=axis, edge_order=edge_order)
    invG = calc_inv_G_value(dxdq, masses)
    G = 1.0 / invG
    if plot:
        plt.plot(q, G)
        plt.xlabel("q")
        plt.ylabel("G")
        if save:
            plt.savefig("gelement.pdf")
        plt.show()
    return G


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


def get_G_matrix(q1, q2, positions, masses, edge_order=2, method="cubic", plot=False, save_op=False, **kwargs):
    print(positions.ndim)
    n_q1, n_q2, n_atoms, ndim = positions.shape

    dx_dq1 = get_dxdq(q1, positions, method=method, axis=0)
    dx_dq2 = get_dxdq(q2, positions, method=method, axis=1)
    invG = np.zeros((n_q1, n_q2, 2, 2))

    invG[:, :, 0, 0] = np.sum(masses[None, None, :, None] * dx_dq1 * dx_dq1, axis=(2, 3))
    invG[:, :, 1, 1] = np.sum(masses[None, None, :, None] * dx_dq2 * dx_dq2, axis=(2, 3))
    invG[:, :, 0, 1] = invG[:, :, 1, 0] = np.sum(masses[None, None, :, None] * dx_dq1 * dx_dq2, axis=(2, 3))

    G = np.linalg.inv(invG)
    if save_op:
        meta = {"class": "OGridMat", "dim":
                [ {"xmin": np.min(q1), "xmax": np.max(q1)},
                  {"xmin": np.min(q2), "xmax": np.max(q2)}
                ] }
        opreader = qdop.FileOP()
        names = []
        for i in range(2):
            for j in range(2):
                gmat_idx = f"{i}{j}"
                fname = f"gmat_{gmat_idx}.op"
                names.append(fname)
                opreader.write(fname, G[:, :, i, j], meta)
        print("G matrix written to: ")
        for name in names:
            print(f"{name:>15s}")
    if plot:
        plot_G_matrix(q1, q2, G, **kwargs)

    return G


def read_G_matrix_operators(names=None, **kwargs):
    if names is None:
        names = []
        for i in range(2):
            for j in range(2):
                name = f"gmat_{i}{j}.op"
                names.append(name)
    opreader = qdop.FileOP()
    comps = []
    meta = None
    for name in names:
        op, meta_read = opreader.read(name)
        if meta is None:
            meta = meta_read
        comps.append(op)
    
    n_q1 = meta["dim"][0].size
    n_q2 = meta["dim"][1].size
    
    q1 = np.linspace(meta["dim"][0].xmin, meta["dim"][0].xmax, n_q1)
    q2 = np.linspace(meta["dim"][1].xmin, meta["dim"][1].xmax, n_q2)
    
    G = np.asarray(comps)               # (4, n_q1, n_q2)
    G = G.reshape(2, 2, n_q1, n_q2)    # (2, 2, n_q1, n_q2)
    G = np.moveaxis(G, (0, 1), (-2, -1))  # (n_q1, n_q2, 2, 2)

    plot_G_matrix(q1, q2, G, **kwargs)

