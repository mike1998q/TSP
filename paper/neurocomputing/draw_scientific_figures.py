#!/usr/bin/env python3
"""Draw DD-Mamba figures in the visual language of DFIR-DETR Fig. 1.

The reference style is translated into reusable scientific-figure primitives:
soft gradient regions, rounded dashed boundaries, shadowed tensor slabs,
vertical gradient module bars, colored main-path arrows, black dashed
auxiliary links, and circular merge operators. Model semantics and reported
numbers remain those of the DD-Mamba manuscript.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".mplconfig")))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)


OUT = Path(__file__).resolve().parent

INK = "#202020"
MUTED = "#60656C"
BORDER = "#AAAEB4"
GRID = "#D8DADF"
PINK = "#EBA4B4"
PURPLE = "#B77AD7"
GREEN = "#05A95A"
GOLD = "#967000"
BLUE_FRONT = "#B8C9EC"
BLUE_SIDE = "#7E91B8"
BLUE_TOP = "#E6EDFA"
MINT_FRONT = "#A9DDD6"
MINT_SIDE = "#7FA99F"
MINT_TOP = "#E7F5DF"
MODULE_TOP = "#F3D0F6"
MODULE_BOTTOM = "#FBFAFB"
PANEL_GREEN_1 = "#EEF7E5"
PANEL_GREEN_2 = "#FFFFFF"
PANEL_BLUE_1 = "#F5F7FC"
PANEL_BLUE_2 = "#DFE9FA"
PANEL_LILAC_1 = "#F8F5FB"
PANEL_LILAC_2 = "#ECE5F5"

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "savefig.facecolor": "white",
        "figure.dpi": 180,
    }
)


def _gradient(c1: str, c2: str, horizontal: bool = True, n: int = 256):
    a = np.asarray(to_rgb(c1))
    b = np.asarray(to_rgb(c2))
    t = np.linspace(0.0, 1.0, n)
    rgb = (1 - t)[:, None] * a + t[:, None] * b
    if horizontal:
        return np.tile(rgb[None, :, :], (2, 1, 1))
    return np.tile(rgb[:, None, :], (1, 2, 1))


def canvas(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def gradient_panel(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    c1: str,
    c2: str,
    title: str,
    *,
    title_x: float | None = None,
):
    shadow = FancyBboxPatch(
        (x + 0.006, y - 0.007),
        w,
        h,
        boxstyle="round,pad=0.006,rounding_size=0.025",
        facecolor="#000000",
        edgecolor="none",
        alpha=0.10,
        zorder=0,
    )
    ax.add_patch(shadow)
    frame = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.006,rounding_size=0.025",
        facecolor="none",
        edgecolor=BORDER,
        linewidth=1.0,
        linestyle=(0, (5, 3)),
        zorder=1,
    )
    ax.add_patch(frame)
    img = ax.imshow(
        _gradient(c1, c2, horizontal=True),
        extent=(x, x + w, y, y + h),
        origin="lower",
        interpolation="bicubic",
        zorder=0.5,
        aspect="auto",
    )
    img.set_clip_path(frame)
    ax.text(
        title_x if title_x is not None else x + w / 2,
        y + 0.018,
        title,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=INK,
        zorder=12,
    )
    return frame


def tensor(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    d: float,
    label: str,
    *,
    palette: str = "blue",
    label_offset: float = 0.024,
    zorder: int = 6,
):
    if palette == "mint":
        front, side, top = MINT_FRONT, MINT_SIDE, MINT_TOP
    else:
        front, side, top = BLUE_FRONT, BLUE_SIDE, BLUE_TOP

    shadow = Polygon(
        [
            (x + 0.006, y - 0.006),
            (x + w + 0.006, y - 0.006),
            (x + w + d + 0.010, y + d - 0.002),
            (x + w + d + 0.010, y + h + d - 0.002),
            (x + d + 0.010, y + h + d - 0.002),
        ],
        closed=True,
        facecolor="black",
        edgecolor="none",
        alpha=0.14,
        zorder=zorder - 2,
    )
    ax.add_patch(shadow)
    front_patch = Rectangle(
        (x, y),
        w,
        h,
        facecolor=front,
        edgecolor="#60656C",
        linewidth=0.8,
        zorder=zorder,
    )
    top_patch = Polygon(
        [(x, y + h), (x + d, y + h + d), (x + w + d, y + h + d), (x + w, y + h)],
        closed=True,
        facecolor=top,
        edgecolor="#60656C",
        linewidth=0.8,
        zorder=zorder + 1,
    )
    side_patch = Polygon(
        [(x + w, y), (x + w + d, y + d), (x + w + d, y + h + d), (x + w, y + h)],
        closed=True,
        facecolor=side,
        edgecolor="#60656C",
        linewidth=0.8,
        zorder=zorder + 1,
    )
    ax.add_patch(front_patch)
    ax.add_patch(top_patch)
    ax.add_patch(side_patch)
    ax.text(
        x + w / 2 + d / 2,
        y - label_offset,
        label,
        ha="center",
        va="top",
        fontsize=7.6,
        color=INK,
        zorder=zorder + 3,
    )
    return {
        "left": (x, y + h / 2),
        "right": (x + w + d, y + h / 2 + d / 2),
        "top": (x + w / 2 + d / 2, y + h + d),
        "bottom": (x + w / 2, y),
        "center": (x + w / 2 + d / 3, y + h / 2 + d / 3),
    }


def module(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    *,
    rotation: int = 90,
    fontsize: float = 8.5,
    c1: str = MODULE_TOP,
    c2: str = MODULE_BOTTOM,
    zorder: int = 8,
):
    shadow = Rectangle(
        (x + 0.006, y - 0.006),
        w,
        h,
        facecolor="black",
        edgecolor="none",
        alpha=0.16,
        zorder=zorder - 2,
    )
    ax.add_patch(shadow)
    frame = Rectangle(
        (x, y),
        w,
        h,
        facecolor="none",
        edgecolor="#777B80",
        linewidth=0.75,
        zorder=zorder,
    )
    ax.add_patch(frame)
    img = ax.imshow(
        _gradient(c1, c2, horizontal=True),
        extent=(x, x + w, y, y + h),
        origin="lower",
        interpolation="bicubic",
        zorder=zorder - 1,
        aspect="auto",
    )
    img.set_clip_path(frame)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        rotation=rotation,
        fontsize=fontsize,
        color=INK,
        zorder=zorder + 1,
    )
    return {
        "left": (x, y + h / 2),
        "right": (x + w, y + h / 2),
        "top": (x + w / 2, y + h),
        "bottom": (x + w / 2, y),
        "center": (x + w / 2, y + h / 2),
    }


def arrow(
    ax,
    start,
    end,
    *,
    color=INK,
    linewidth=1.0,
    dashed=False,
    rad=0.0,
    mutation_scale=9,
    zorder=5,
):
    a = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        linestyle=(0, (5, 3)) if dashed else "-",
        # Keep every connector geometrically straight.  The ``rad`` argument
        # remains in the public helper signature so older call sites still
        # work, but curvature is intentionally disabled for paper figures.
        connectionstyle="arc3,rad=0",
        shrinkA=1.5,
        shrinkB=1.5,
        zorder=zorder,
    )
    ax.add_patch(a)
    return a


def elbow(
    ax,
    points,
    *,
    color=INK,
    linewidth=1.0,
    dashed=False,
    mutation_scale=9,
    zorder=4,
):
    style = (0, (5, 3)) if dashed else "-"
    for i in range(len(points) - 2):
        p0, p1 = points[i], points[i + 1]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, linewidth=linewidth, linestyle=style, zorder=zorder)
    arrow(
        ax,
        points[-2],
        points[-1],
        color=color,
        linewidth=linewidth,
        dashed=dashed,
        mutation_scale=mutation_scale,
        zorder=zorder,
    )


def ortho_bus(ax, start, end, bus_x, **kw):
    """Right-angle 'bus' connector with true 90-degree corners: horizontal from
    ``start`` to ``bus_x``, vertical to ``end``'s height, then horizontal into
    ``end``. Use for connections that change both row and column."""
    (sx, sy), (ex, ey) = start, end
    elbow(ax, [start, (bus_x, sy), (bus_x, ey), end], **kw)


def ortho_L(ax, start, end, *, first="h", **kw):
    """Right-angle L connector. ``first='h'``: horizontal then vertical into
    ``end`` (enter from the side); ``first='v'``: vertical then horizontal into
    ``end`` (enter from top/bottom)."""
    (sx, sy), (ex, ey) = start, end
    mid = (ex, sy) if first == "h" else (sx, ey)
    elbow(ax, [start, mid, end], **kw)


def op_node(ax, x: float, y: float, text: str, *, edge=INK, radius=0.014, fontsize=8.5):
    circle = Circle((x, y), radius, facecolor="white", edgecolor=edge, linewidth=0.9, zorder=10)
    ax.add_patch(circle)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=INK, zorder=11)
    return {
        "left": (x - radius, y),
        "right": (x + radius, y),
        "top": (x, y + radius),
        "bottom": (x, y - radius),
        "center": (x, y),
    }


def label_box(ax, x, y, w, h, text, *, face="#F2F5FB", edge="#7A7E84", fontsize=8.5):
    shadow = Rectangle((x + 0.005, y - 0.005), w, h, facecolor="black", edgecolor="none", alpha=0.10, zorder=6)
    ax.add_patch(shadow)
    rect = Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=0.75, zorder=7)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=INK, zorder=8)
    return {
        "left": (x, y + h / 2),
        "right": (x + w, y + h / 2),
        "top": (x + w / 2, y + h),
        "bottom": (x + w / 2, y),
        "center": (x + w / 2, y + h / 2),
    }


def save(fig, stem: str):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(OUT / f"{stem}.png", dpi=260, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def draw_architecture():
    fig, ax = canvas(12.0, 6.85)

    gradient_panel(ax, 0.012, 0.14, 0.175, 0.76, PANEL_GREEN_1, PANEL_GREEN_2, "Preprocessing")
    gradient_panel(
        ax,
        0.198,
        0.535,
        0.585,
        0.365,
        PANEL_BLUE_1,
        PANEL_BLUE_2,
        "Time branch",
        title_x=0.360,
    )
    gradient_panel(
        ax,
        0.198,
        0.14,
        0.585,
        0.365,
        PANEL_GREEN_1,
        PANEL_GREEN_2,
        "Frequency branch",
        title_x=0.360,
    )
    gradient_panel(ax, 0.794, 0.14, 0.194, 0.76, PANEL_BLUE_1, PANEL_BLUE_2, "Forecast fusion")

    # Input and normalization.
    x_in = tensor(ax, 0.035, 0.49, 0.052, 0.235, 0.018, r"$\mathbf{X}$" + "\n" + r"$L\times C$", palette="mint")
    rev = module(ax, 0.128, 0.515, 0.030, 0.205, "RevIN")
    arrow(ax, x_in["right"], rev["left"], color=PURPLE, linewidth=1.6, mutation_scale=11)
    ax.text(0.098, 0.645, r"$\boldsymbol{\mu},\boldsymbol{\sigma}$", ha="center", va="bottom", fontsize=7.8, color=MUTED)

    # Time branch: decomposition followed by anchor and correction paths.
    dec = module(ax, 0.225, 0.650, 0.028, 0.150, "Decomp.")
    t_repr = tensor(ax, 0.280, 0.662, 0.042, 0.135, 0.014, r"$\mathbf{X}_s,\mathbf{X}_t$" + "\n" + r"$L\times C$", palette="blue")
    ortho_bus(ax, rev["right"], dec["left"], 0.190, color=PURPLE, linewidth=1.5, mutation_scale=10)
    arrow(ax, dec["right"], t_repr["left"], color=PURPLE, linewidth=1.5, mutation_scale=10)

    dlinear = module(ax, 0.365, 0.731, 0.026, 0.115, "DLinear", fontsize=7.8)
    ylin = tensor(ax, 0.425, 0.735, 0.036, 0.090, 0.012, r"$\mathbf{Y}_{\mathrm{lin}}$" + "\n" + r"$H\times C$", palette="blue")
    ortho_bus(ax, t_repr["right"], dlinear["left"], 0.345, color=PINK, linewidth=1.4)
    arrow(ax, dlinear["right"], ylin["left"], color=PINK, linewidth=1.4)

    mamba = module(ax, 0.365, 0.575, 0.026, 0.118, "Mamba", fontsize=7.8)
    f_t = tensor(ax, 0.425, 0.585, 0.040, 0.090, 0.012, r"$\mathbf{f}_{\mathrm{t}}$" + "\n" + r"$C\times d$", palette="blue")
    mixer_t = module(ax, 0.505, 0.580, 0.027, 0.122, "Mixer", fontsize=7.8)
    head_t = module(ax, 0.575, 0.580, 0.027, 0.122, r"$\phi_{\mathrm{t}}$", fontsize=9.5)
    ortho_bus(ax, t_repr["right"], mamba["left"], 0.345, color=PURPLE, linewidth=1.4, mutation_scale=10)
    arrow(ax, mamba["right"], f_t["left"], color=PURPLE, linewidth=1.4)
    arrow(ax, f_t["right"], mixer_t["left"], color=GREEN, linewidth=1.3)
    arrow(ax, mixer_t["right"], head_t["left"], color=GREEN, linewidth=1.3)

    sum_t = op_node(ax, 0.670, 0.630, "+")
    yt = tensor(ax, 0.716, 0.585, 0.035, 0.095, 0.011, r"$\mathbf{Y}_{\mathrm{t}}$" + "\n" + r"$H\times C$", palette="blue")
    ortho_L(ax, ylin["right"], sum_t["top"], first="h", color=INK, dashed=True)
    arrow(ax, head_t["right"], sum_t["left"], color=INK, dashed=True)
    arrow(ax, sum_t["right"], yt["left"], color=PURPLE, linewidth=1.4)
    ax.text(0.591, 0.565, "zero-init correction", ha="center", va="top", fontsize=7.2, color=MUTED)

    # Frequency branch.
    rfft = module(ax, 0.225, 0.245, 0.028, 0.155, "rFFT")
    z = tensor(ax, 0.280, 0.263, 0.042, 0.120, 0.014, r"$\mathbf{Z}$" + "\n" + r"$C\times F$", palette="mint")
    ortho_bus(ax, rev["right"], rfft["left"], 0.190, color=GREEN, linewidth=1.5)
    arrow(ax, rfft["right"], z["left"], color=GREEN, linewidth=1.5)

    spec = module(ax, 0.365, 0.334, 0.026, 0.116, "Spectral", fontsize=7.5)
    f_f = tensor(ax, 0.425, 0.340, 0.040, 0.088, 0.012, r"$\mathbf{f}_{\mathrm{f}}$" + "\n" + r"$C\times d$", palette="mint")
    mixer_f = module(ax, 0.505, 0.329, 0.027, 0.122, "Mixer", fontsize=7.8)
    head_f = module(ax, 0.575, 0.329, 0.027, 0.122, r"$\phi_{\mathrm{f}}$", fontsize=9.5)
    ortho_bus(ax, z["right"], spec["left"], 0.345, color=GREEN, linewidth=1.3)
    arrow(ax, spec["right"], f_f["left"], color=GREEN, linewidth=1.3)
    arrow(ax, f_f["right"], mixer_f["left"], color=GREEN, linewidth=1.3)
    arrow(ax, mixer_f["right"], head_f["left"], color=GREEN, linewidth=1.3)

    fits = module(ax, 0.365, 0.174, 0.026, 0.115, "FITS map", fontsize=7.4)
    yfits = tensor(ax, 0.425, 0.183, 0.036, 0.086, 0.012, r"$\mathbf{Y}_{\mathrm{FITS}}$" + "\n" + r"$H\times C$", palette="mint")
    ortho_bus(ax, z["right"], fits["left"], 0.345, color=GOLD, linewidth=1.2)
    arrow(ax, fits["right"], yfits["left"], color=GOLD, linewidth=1.2)

    sum_f = op_node(ax, 0.670, 0.390, "+")
    yf = tensor(ax, 0.716, 0.343, 0.035, 0.095, 0.011, r"$\mathbf{Y}_{\mathrm{f}}$" + "\n" + r"$H\times C$", palette="mint")
    arrow(ax, head_f["right"], sum_f["left"], color=INK, dashed=True)
    ortho_L(ax, yfits["right"], sum_f["bottom"], first="h", color=INK, dashed=True)
    arrow(ax, sum_f["right"], yf["left"], color=GREEN, linewidth=1.4)
    ax.text(0.590, 0.312, "zero-init forecast", ha="center", va="bottom", fontsize=7.2, color=MUTED)

    # Fusion head.
    gate = module(ax, 0.830, 0.650, 0.030, 0.160, "Gate $g$", fontsize=8.2)
    fuse = op_node(ax, 0.894, 0.512, r"$\Sigma_g$", edge=GOLD, radius=0.017, fontsize=8.2)
    denorm = module(ax, 0.875, 0.320, 0.034, 0.130, "De-norm", fontsize=8.2, c1="#E3F4EE", c2="#FBFEFD")
    yout = tensor(ax, 0.932, 0.245, 0.030, 0.160, 0.010, r"$\widehat{\mathbf{Y}}$" + "\n" + r"$H\times C$", palette="blue")

    # Both branch forecasts merge into the gate-weighted sum along one vertical
    # bus, entering the sum node from the left.
    ortho_bus(ax, yt["right"], fuse["left"], 0.775, color=PURPLE, linewidth=1.5)
    ortho_bus(ax, yf["right"], fuse["left"], 0.775, color=GREEN, linewidth=1.5)
    # The gate reads both branch features; route the taps clear of the forecast
    # columns (over the top for the time feature, under the bottom for the
    # frequency feature) so no line crosses a tensor or module.
    elbow(ax, [f_t["top"], (f_t["top"][0], 0.884), (gate["top"][0], 0.884), gate["top"]],
          color=INK, dashed=True)
    elbow(ax, [f_f["bottom"], (f_f["bottom"][0], 0.104), (gate["bottom"][0], 0.104), gate["bottom"]],
          color=INK, dashed=True)
    ortho_L(ax, gate["right"], fuse["top"], first="v", color=GOLD, linewidth=1.2)
    arrow(ax, fuse["bottom"], denorm["top"], color=GOLD, linewidth=1.3)
    ortho_L(ax, denorm["right"], yout["left"], first="h", color=INK, dashed=True)
    ax.text(0.905, 0.487, r"$g\mathbf{Y}_{\mathrm{t}}+(1-g)\mathbf{Y}_{\mathrm{f}}$", ha="left", va="top", fontsize=7.2, color=MUTED)

    # Legend, echoing the reference figure's bottom-right key.
    ax.text(0.020, 0.076, "Legend", ha="left", va="center", fontsize=9.5, fontweight="bold")
    arrow(ax, (0.085, 0.077), (0.125, 0.077), color=PURPLE, linewidth=1.7)
    ax.text(0.131, 0.077, "time path", ha="left", va="center", fontsize=7.8)
    arrow(ax, (0.222, 0.077), (0.262, 0.077), color=GREEN, linewidth=1.5)
    ax.text(0.268, 0.077, "frequency path", ha="left", va="center", fontsize=7.8)
    arrow(ax, (0.392, 0.077), (0.432, 0.077), color=GOLD, linewidth=1.3)
    ax.text(0.438, 0.077, "gate / spectral map", ha="left", va="center", fontsize=7.8)
    arrow(ax, (0.590, 0.077), (0.630, 0.077), color=INK, dashed=True, linewidth=1.0)
    ax.text(0.636, 0.077, "auxiliary connection", ha="left", va="center", fontsize=7.8)
    module(ax, 0.790, 0.058, 0.020, 0.040, "", rotation=0)
    ax.text(0.818, 0.077, "learned module", ha="left", va="center", fontsize=7.8)

    save(fig, "fig_architecture")


def draw_selective_blocks():
    fig, ax = canvas(12.0, 6.0)
    gradient_panel(ax, 0.015, 0.09, 0.475, 0.84, PANEL_BLUE_1, PANEL_BLUE_2, "Causal Mamba layer")
    gradient_panel(ax, 0.510, 0.09, 0.475, 0.84, PANEL_GREEN_1, PANEL_GREEN_2, "Bidirectional variate mixer")

    # Panel (a): a compact circuit using the reference's tensor slabs and bars.
    inp = tensor(ax, 0.040, 0.590, 0.038, 0.205, 0.014, r"$u$" + "\n" + r"$L\times d_{\mathrm{in}}$", palette="blue")
    norm = module(ax, 0.112, 0.610, 0.025, 0.165, "RMSNorm", fontsize=7.5)
    iproj = module(ax, 0.165, 0.610, 0.025, 0.165, "in-proj", fontsize=7.5)
    uproj = tensor(ax, 0.220, 0.625, 0.035, 0.125, 0.012, r"$u_p$" + "\n" + r"$L\times d$", palette="blue")
    conv = module(ax, 0.295, 0.620, 0.025, 0.160, "Conv1d", fontsize=7.5)
    content = tensor(ax, 0.350, 0.635, 0.035, 0.115, 0.012, r"$x_k$", palette="blue")
    scan = module(ax, 0.425, 0.595, 0.028, 0.205, "Selective scan", fontsize=7.1)

    arrow(ax, inp["right"], norm["left"], color=PURPLE, linewidth=1.5)
    arrow(ax, norm["right"], iproj["left"], color=PURPLE, linewidth=1.5)
    arrow(ax, iproj["right"], uproj["left"], color=PURPLE, linewidth=1.5)
    arrow(ax, uproj["right"], conv["left"], color=PINK, linewidth=1.4)
    arrow(ax, conv["right"], content["left"], color=PINK, linewidth=1.4)
    arrow(ax, content["right"], scan["left"], color=GREEN, linewidth=1.3)

    xproj = module(ax, 0.334, 0.360, 0.027, 0.150, r"$x$-proj", fontsize=7.5)
    params = tensor(ax, 0.392, 0.377, 0.035, 0.105, 0.012, r"$\Delta_k,B_k,C_k$", palette="mint")
    ortho_L(ax, content["bottom"], xproj["top"], first="v", color=GREEN, linewidth=1.2)
    arrow(ax, xproj["right"], params["left"], color=GREEN, linewidth=1.2)
    arrow(ax, params["right"], scan["bottom"], color=GREEN, linewidth=1.2)

    silu = module(ax, 0.220, 0.335, 0.027, 0.150, "SiLU gate", fontsize=7.4)
    mul = op_node(ax, 0.400, 0.270, r"$\odot$", edge=PURPLE, radius=0.015)
    outproj = module(ax, 0.345, 0.145, 0.028, 0.095, "out-proj", fontsize=7.4)
    add = op_node(ax, 0.265, 0.190, "+", edge=PURPLE, radius=0.015)
    out = tensor(ax, 0.120, 0.145, 0.040, 0.100, 0.012, "corrected\nfeatures", palette="blue")

    ortho_L(ax, uproj["bottom"], silu["top"], first="v", color=PURPLE, linewidth=1.2)
    ortho_L(ax, silu["right"], mul["left"], first="v", color=PURPLE, linewidth=1.2)
    ortho_bus(ax, scan["bottom"], mul["right"], 0.462, color=INK, dashed=True)
    ortho_L(ax, mul["bottom"], outproj["right"], first="v", color=INK, dashed=True)
    arrow(ax, outproj["left"], add["right"], color=PURPLE, linewidth=1.2)
    ortho_L(ax, inp["bottom"], add["left"], first="v", color=INK, dashed=True)
    arrow(ax, add["left"], out["right"], color=PURPLE, linewidth=1.3)
    ax.text(
        0.372,
        0.855,
        r"$\mathbf{h}_k=\bar{\mathbf{A}}_k\mathbf{h}_{k-1}+\bar{\mathbf{B}}_ku_k$",
        ha="center",
        va="center",
        fontsize=7.0,
        color=MUTED,
    )

    # Panel (b): bidirectional scan and two residual refinements.
    tok = tensor(ax, 0.535, 0.600, 0.040, 0.190, 0.014, r"$\mathbf{F}$" + "\n" + r"$C\times d$", palette="mint")
    fwd = module(ax, 0.625, 0.680, 0.026, 0.135, r"Mamba $\rightarrow$", fontsize=7.1)
    bwd = module(ax, 0.625, 0.430, 0.026, 0.155, r"flip-Mamba-flip", fontsize=6.7)
    ft = tensor(ax, 0.690, 0.690, 0.035, 0.105, 0.012, "forward", palette="mint")
    fb = tensor(ax, 0.690, 0.450, 0.035, 0.105, 0.012, "backward", palette="mint")
    merge = op_node(ax, 0.790, 0.615, "+", edge=GREEN, radius=0.016)
    res1 = op_node(ax, 0.840, 0.615, "+", edge=PURPLE, radius=0.016)
    nrm = module(ax, 0.875, 0.535, 0.027, 0.155, "RMSNorm", fontsize=7.4)
    ffn = module(ax, 0.925, 0.535, 0.027, 0.155, "FFN", fontsize=8.0)

    ortho_bus(ax, tok["right"], fwd["left"], 0.600, color=GREEN, linewidth=1.3)
    ortho_bus(ax, tok["right"], bwd["left"], 0.600, color=GREEN, linewidth=1.3)
    arrow(ax, fwd["right"], ft["left"], color=GREEN, linewidth=1.3)
    arrow(ax, bwd["right"], fb["left"], color=GREEN, linewidth=1.3)
    ortho_L(ax, ft["right"], merge["top"], first="h", color=INK, dashed=True)
    ortho_L(ax, fb["right"], merge["bottom"], first="h", color=INK, dashed=True)
    arrow(ax, merge["right"], res1["left"], color=PURPLE, linewidth=1.3)
    elbow(ax, [tok["top"], (tok["top"][0], 0.865), (res1["top"][0], 0.865), res1["top"]], color=INK, dashed=True)
    arrow(ax, res1["right"], nrm["left"], color=PURPLE, linewidth=1.3)
    arrow(ax, nrm["right"], ffn["left"], color=PURPLE, linewidth=1.3)

    res2 = op_node(ax, 0.880, 0.360, "+", edge=PURPLE, radius=0.016)
    out2 = tensor(ax, 0.925, 0.175, 0.038, 0.120, 0.012, "mixed\nfeatures", palette="mint")
    ortho_L(ax, ffn["bottom"], res2["right"], first="v", color=PURPLE, linewidth=1.2)
    ortho_L(ax, res1["bottom"], res2["left"], first="v", color=INK, dashed=True)
    ortho_L(ax, res2["bottom"], out2["left"], first="v", color=PURPLE, linewidth=1.3)

    save(fig, "fig_selective_blocks")


def _panel_background(fig, x, y, w, h, c1, c2):
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    gradient_panel(ax, x, y, w, h, c1, c2, "", title_x=x + w / 2)
    return ax


def draw_dispersion():
    fig = plt.figure(figsize=(11.3, 4.2))
    _panel_background(fig, 0.025, 0.11, 0.285, 0.80, PANEL_GREEN_1, PANEL_GREEN_2)
    _panel_background(fig, 0.355, 0.11, 0.285, 0.80, PANEL_BLUE_1, PANEL_BLUE_2)
    _panel_background(fig, 0.685, 0.11, 0.290, 0.80, PANEL_LILAC_1, PANEL_LILAC_2)

    # (a) Range plot.
    ax1 = fig.add_axes([0.060, 0.305, 0.215, 0.455])
    ax1.set_xlim(0, 0.6)
    ax1.set_ylim(0, 1)
    ax1.set_yticks([])
    ax1.grid(axis="x", color=GRID, linewidth=0.7)
    ax1.set_axisbelow(True)
    ax1.axvspan(0.20, 0.48, ymin=0.30, ymax=0.70, color="#CFE2C8", alpha=0.55, zorder=1)
    ax1.plot([0.20, 0.48], [0.50, 0.50], color=GREEN, linewidth=3.2, solid_capstyle="round", zorder=3)
    ax1.scatter([0.20, 0.48], [0.50, 0.50], s=70, color=[MINT_SIDE, MINT_SIDE], edgecolor="#60656C", linewidth=0.6, zorder=4)
    ax1.text(0.20, 0.60, "0.20", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.text(0.48, 0.60, "0.48", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.text(0.34, 0.36, "reported span across\nprobed datasets/channels", ha="center", va="top", fontsize=8, color=MUTED)
    ax1.set_xlabel("Coefficient of variation of rolling std", labelpad=6)
    for s in ("top", "right", "left"):
        ax1.spines[s].set_visible(False)
    ax1.spines["bottom"].set_color(BORDER)

    # (b) RevIN effect.
    ax2 = fig.add_axes([0.392, 0.305, 0.215, 0.455])
    vals = [-0.029, 0.100]
    names = ["Solar", "Traffic"]
    colors = [PINK, MINT_SIDE]
    bars = ax2.barh(names, vals, color=colors, edgecolor="#6F7377", linewidth=0.7, height=0.48, zorder=3)
    for bar in bars:
        ax2.add_patch(
            Rectangle(
                (bar.get_x() + 0.004, bar.get_y() - 0.025),
                bar.get_width(),
                bar.get_height(),
                facecolor="black",
                edgecolor="none",
                alpha=0.10,
                zorder=1,
            )
        )
    ax2.axvline(0, color=INK, linewidth=0.9, zorder=4)
    ax2.set_xlim(-0.05, 0.12)
    ax2.set_xlabel(r"$\Delta$MSE = off $-$ on", labelpad=6)
    ax2.grid(axis="x", color=GRID, linewidth=0.7)
    ax2.set_axisbelow(True)
    for bar, value in zip(bars, vals):
        ax2.text(
            value * 0.88,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.3f}",
            ha="right" if value > 0 else "left",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="white",
            zorder=5,
        )
    ax2.text(0.112, 1.27, "RevIN helps", ha="right", va="center", fontsize=8, color="#147A4C")
    ax2.text(-0.047, -0.27, "RevIN hurts", ha="left", va="center", fontsize=8, color="#B74455")
    for s in ("top", "right", "left"):
        ax2.spines[s].set_visible(False)
    ax2.spines["bottom"].set_color(BORDER)
    ax2.tick_params(axis="y", length=0)

    # (c) KPSS and autocorrelation.
    ax3 = fig.add_axes([0.720, 0.305, 0.220, 0.455])
    names3 = ["Solar", "Traffic", "ETTh1"]
    cols3 = [PINK, MINT_SIDE, BLUE_SIDE]
    bars3 = ax3.bar(names3, [1, 1, 1], color=cols3, edgecolor="#6F7377", linewidth=0.7, width=0.54, zorder=3)
    for bar in bars3:
        ax3.add_patch(
            Rectangle(
                (bar.get_x() + 0.03, -0.02),
                bar.get_width(),
                1.0,
                facecolor="black",
                edgecolor="none",
                alpha=0.08,
                zorder=1,
            )
        )
        ax3.text(bar.get_x() + bar.get_width() / 2, 1.035, "100%", ha="center", va="bottom", fontsize=8.8, fontweight="bold")
    ax3.set_ylim(0, 1.16)
    ax3.set_ylabel("KPSS rejection fraction")
    ax3.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", "25%", "50%", "75%", "100%"])
    ax3.grid(axis="y", color=GRID, linewidth=0.7)
    ax3.set_axisbelow(True)
    for s in ("top", "right"):
        ax3.spines[s].set_visible(False)
    ax3.spines["bottom"].set_color(BORDER)
    ax3.spines["left"].set_color(BORDER)

    # Bottom group labels, as in the reference architecture figure.
    overlay = fig.add_axes([0, 0, 1, 1])
    overlay.set_xlim(0, 1)
    overlay.set_ylim(0, 1)
    overlay.axis("off")
    overlay.text(0.1675, 0.130, "(a) Rolling-scale variability", ha="center", va="bottom", fontsize=11, fontweight="bold")
    overlay.text(0.4975, 0.130, "(b) RevIN removal effect", ha="center", va="bottom", fontsize=11, fontweight="bold")
    overlay.text(0.8300, 0.130, "(c) Rolling-std stationarity", ha="center", va="bottom", fontsize=11, fontweight="bold")
    overlay.text(
        0.8300,
        0.820,
        "Ljung-Box: strong autocorrelation\nin every tested rolling-std series",
        ha="center",
        va="top",
        fontsize=7.7,
        color=MUTED,
    )

    save(fig, "fig_dispersion")


if __name__ == "__main__":
    draw_architecture()
    draw_selective_blocks()
    draw_dispersion()
    print("Generated R2-style figures:")
    for name in ("fig_architecture", "fig_selective_blocks", "fig_dispersion"):
        print(f"  {name}.pdf")
        print(f"  {name}.png")
