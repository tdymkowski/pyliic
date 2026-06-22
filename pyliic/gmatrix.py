#! /usr/bin/env python3
import numpy as np
import qdio.qd_file_op as qdop
from scipy.interpolate import CubicSpline, PchipInterpolator
import matplotlib.pyplot as plt
from .data import AMU2AU


def reshape_positions_for_gmat(q1_flat, q2_flat, positions):
    q1_flat = np.asarray(q1_flat)
    q2_flat = np.asarray(q2_flat)
    positions = np.asarray(positions)

    q1_unique = np.unique(q1_flat)
    q2_unique = np.unique(q2_flat)

    n_q1 = len(q1_unique)
    n_q2 = len(q2_unique)
    n_geoms, n_atoms, ndim = positions.shape

    if n_geoms != n_q1 * n_q2:
        raise ValueError(
            f"Cannot reshape: got {n_geoms} geometries, but "
            f"{n_q1} unique q1 values × {n_q2} unique q2 values = "
            f"{n_q1 * n_q2} grid points."
        )

    positions_grid = np.zeros((n_q1, n_q2, n_atoms, ndim))

    for k, (q1, q2) in enumerate(zip(q1_flat, q2_flat)):
        i = np.where(q1_unique == q1)[0][0]
        j = np.where(q2_unique == q2)[0][0]
        positions_grid[i, j] = positions[k]

    return q1_unique, q2_unique, positions_grid


def _sort_q_and_positions(q, positions, axis=0):
    order = np.argsort(q)
    q = q[order]
    if any(np.diff(q) <= 0.):
        raise ValueError(f"Internal coordinate must be striclty increasing sequence!")
    x = np.take(positions, order, axis=axis)
    return q, x


def get_dxdq_fd(q, positions, edge_order=2, axis=0):
    q, x = _sort_q_and_positions(q, positions, axis=axis)
    dxdq = np.gradient(x, q, axis=axis, edge_order=edge_order)
    return dxdq


def get_dxdq_spline(q, positions, spline_type="cubic", axis=0, spline_kwargs={}):
    spline_methods = {"cubic": CubicSpline, "pchip": PchipInterpolator}
    q, x = _sort_q_and_positions(q, positions, axis=axis)
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
    print(invG/ AMU2AU)
    G = 1.0 / invG
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(q, G)
        ax1.set_xlabel("q")
        ax1.set_ylabel("G")
        ax2.plot(q, invG / AMU2AU)
        ax2.set_xlabel("q")
        ax2.set_ylabel("invG")
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



    if invG is not None:
        components = [
            (0, 0, "invG_rr"),
            (0, 1, "invG_rphi"),
            (1, 0, "invG_phir"),
            (1, 1, "invG_phiphi"),
        ]

        fig2, axes2 = plt.subplots(2, 2, figsize=kwargs.get("figsize", (10, 8)))
        for ax, (a, b, title) in zip(axes2.flatten(), components):
            contour2 = ax.contourf(R, P, invG[:, :, a, b], levels=levels, cmap=cmap)
            ax.set_xlabel(q1_label)
            ax.set_ylabel(q2_label)
            fig2.colorbar(contour2, ax=ax, label=title)
        fig2.tight_layout()

    plt.show()


def get_G_matrix(q1, q2, positions, masses,
                 edge_order=2,
                 method="cubic",
                 plot=False,
                 plot_invg=False,
                 save_op=False,
                 **kwargs):
    n_q1, n_q2, n_atoms, ndim = positions.shape

    dx_dq1 = get_dxdq(q1,
                      positions,
                      method=method,
                      axis=0,
                      edge_order=edge_order)
    dx_dq2 = get_dxdq(q2,
                      positions,
                      method=method,
                      axis=1,
                      edge_order=edge_order)
    invG = np.zeros((n_q1, n_q2, 2, 2))

    invG[:, :, 0, 0] = np.sum(masses[None, None, :, None] * dx_dq1 * dx_dq1,
                              axis=(2, 3))
    invG[:, :, 1, 1] = np.sum(masses[None, None, :, None] * dx_dq2 * dx_dq2,
                              axis=(2, 3))
    invG[:, :, 0, 1] = invG[:, :, 1, 0] = np.sum(masses[None, None, :, None] *
                                                 dx_dq1 * dx_dq2, axis=(2, 3))

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
    if plot and not plot_invg:
        plot_G_matrix(q1, q2, G, **kwargs)
    if plot_invg:
        plot_G_matrix(q1, q2, G, invG=invG, **kwargs)

    return G, invG


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
