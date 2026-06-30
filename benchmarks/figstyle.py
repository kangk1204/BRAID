"""Shared publication figure style for the BRAID paper figure set.

Call ``figstyle.apply()`` once at the top of every figure-generating script so the
whole set shares one consistent, vector-editable, colorblind-aware look. This module
only *styles* — the data for every figure still comes from each script's own
committed source artifact, never from here.

Key wins applied uniformly:
  * editable vector text in SVG/PDF (matplotlib otherwise outlines text to paths,
    which an editor cannot fix);
  * a single sans-serif family, white background, tight bbox, 300-dpi raster;
  * minimal (top/right-off) spines for a clean publication look.

It deliberately does NOT set figure size or base font size, so each script keeps its
own working layout — apply() is layout-safe to drop into an existing script.
"""

from __future__ import annotations

# Okabe-Ito: colorblind-safe and grayscale-distinguishable. Use where a script picks
# categorical colors, so the set is accessible.
OKABE_ITO = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442", "#000000",
]

# Layout targets (inches): single ~3.5 (89 mm), 1.5-col ~5.0, double ~7.2 (183 mm).
WIDTH_SINGLE = 3.5
WIDTH_ONEHALF = 5.0
WIDTH_DOUBLE = 7.2


def apply() -> None:
    """Set the house rcParams (editable vector text, white bg, clean spines)."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "svg.fonttype": "none",          # keep SVG text editable, not outlined to paths
        "pdf.fonttype": 42,              # embed editable TrueType in PDF
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })


def save_all(fig, stem: str) -> None:
    """Write png (300 dpi raster) + svg + pdf (vector) from one figure."""
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{stem}.{ext}")
    print(f"wrote {stem}.png/.svg/.pdf")


# --- Design system (shared across the BRAID figure set) ----------------------
# One saturated accent for the method of interest (BRAID); everything it is
# compared against is desaturated, so colour carries meaning, not decoration.
# This is what stops the set looking like a generic rainbow default.
HERO = "#08519C"                 # BRAID: deep, saturated blue
NATIVE = "#B6B9BC"               # a comparator's native/uncalibrated bar (muted gray)
NOMINAL = "#2E7D32"              # nominal-coverage guide (green)
INK = "#222222"
MUTED = {                        # desaturated per-comparator hues (forest/scatter)
    "MAJIQ": "#E2B07A",
    "betAS": "#86BBA8",
    "rMATS": "#C7A6BE",
    "SUPPA2": "#9C8FBF",
}


def hero_color(label: str) -> str:
    """Deep accent for BRAID, a muted hue for a named comparator."""
    if "BRAID" in label:
        return HERO
    return MUTED.get(label, NATIVE)


def nominal_guide(ax, level: float = 0.95, axis: str = "x") -> None:
    """Nominal-coverage line plus a light 'at-or-better' shaded band."""
    if axis == "x":
        ax.axvspan(level, 1.02, color=NOMINAL, alpha=0.06, zorder=0)
        ax.axvline(level, color=NOMINAL, ls=(0, (4, 3)), lw=1.1, zorder=1)
    else:
        ax.axhspan(level, 1.02, color=NOMINAL, alpha=0.06, zorder=0)
        ax.axhline(level, color=NOMINAL, ls=(0, (4, 3)), lw=1.1, zorder=1)


def panel_letter(ax, letter: str, x: float = -0.02, y: float = 1.10) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top", ha="right")
