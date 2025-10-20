# IBD over time: gray base points, then color by IBD as dates pass.
# Requirements: pandas, numpy, matplotlib, pillow (for GIF)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize
from pathlib import Path
import os

def over_time(
    df: pd.DataFrame,
    x_col: str = "Longitude_Lambert93",
    y_col: str = "Latitude_Lambert93",
    date_col: str = "Date_SamplingOperation",
    value_col: str = "IBD",
    max_frames: int = 60,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """
    Create:
      (1) Static PNG: all sites gray + colored overlay by IBD.
      (2) Animated GIF: points become colored by IBD as date increases.

    Rules:
      - Drops rows with missing coords or unparsable dates.
      - IBD NaN stays gray forever.
      - Color scale clipped to [vmin, vmax].

    Returns dict with paths and frame count.
    """
    # create visualizations directory if not exists
    os.makedirs("visualizations", exist_ok=True)
    out_gif = Path("visualizations") / f"{value_col}_over_time.gif"
    out_png = Path("visualizations") / f"{value_col}_over_time_static.png"

    df = df.copy()

    if not vmin:
        vmin = df[value_col].min()
    if not vmax:
        vmax = df[value_col].max()

    # Parse dates safely and drop rows without date/coords
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[x_col, y_col, date_col])

    # Base coordinates
    X = df[x_col].to_numpy()
    Y = df[y_col].to_numpy()

    # Valid values for coloring
    val_num = pd.to_numeric(df[value_col], errors="coerce")
    has_val = np.isfinite(val_num.to_numpy())
    Xv = df.loc[has_val, x_col].to_numpy()
    Yv = df.loc[has_val, y_col].to_numpy()
    V  = val_num.loc[has_val].to_numpy()
    dvals = df.loc[has_val, date_col].to_numpy()

    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

    def extent_with_margin(x, y, m=0.02):
        xmin, xmax = np.nanmin(x), np.nanmax(x)
        ymin, ymax = np.nanmin(y), np.nanmax(y)
        dx, dy = xmax - xmin, ymax - ymin
        return (xmin - m*dx, xmax + m*dx, ymin - m*dy, ymax + m*dy)

    xmin, xmax, ymin, ymax = extent_with_margin(X, Y)

    # ---------- Static PNG ----------
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.scatter(X, Y, s=10, color="0.7", alpha=0.7, linewidths=0, label="All sites")
    sc = ax.scatter(Xv, Yv, s=12, c=V, norm=norm, linewidths=0)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(f"{value_col} [{vmin}–{vmax}]")
    ax.set_xlabel(x_col); ax.set_ylabel(y_col)
    ax.set_title(f"{value_col} colored; all locations in gray")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    # ---------- Animated GIF ----------
    dates = np.sort(df[date_col].unique())
    if len(dates) > max_frames:
        idx = np.linspace(0, len(dates)-1, max_frames).astype(int)
        dates = dates[idx]

    fig2, ax2 = plt.subplots(figsize=(7.5, 7.5))
    ax2.scatter(X, Y, s=10, color="0.7", alpha=0.7, linewidths=0)  # base gray
    sc_dyn = ax2.scatter([], [], s=12, c=[], norm=norm, linewidths=0)  # dynamic colored
    cbar2 = plt.colorbar(sc_dyn, ax=ax2)
    cbar2.set_label(f"{value_col} [{vmin}–{vmax}]")
    ax2.set_xlabel(x_col); ax2.set_ylabel(y_col)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlim(xmin, xmax); ax2.set_ylim(ymin, ymax)
    txt = ax2.text(0.02, 0.98, "", transform=ax2.transAxes, va="top", ha="left")

    def init():
        sc_dyn.set_offsets(np.empty((0, 2)))
        sc_dyn.set_array(np.array([]))
        txt.set_text("")
        ax2.set_title(f"{value_col} over time")
        return (sc_dyn, txt)

    def update(i):
        t = dates[i]
        mask = dvals <= t
        offs = np.column_stack([Xv[mask], Yv[mask]])
        sc_dyn.set_offsets(offs)
        sc_dyn.set_array(V[mask])
        txt.set_text(f"Date ≤ {pd.to_datetime(t).date()}  |  colored: {mask.sum()} / {len(V)}")
        return (sc_dyn, txt)

    anim = FuncAnimation(fig2, update, frames=len(dates), init_func=init, blit=True, interval=100)

    saved_gif = True
    try:
        anim.save(out_gif, writer=PillowWriter(fps=10))
    except Exception:
        saved_gif = False
    plt.close(fig2)

    return {"png": str(out_png), "gif": str(out_gif) if saved_gif else None, "frames": len(dates)}