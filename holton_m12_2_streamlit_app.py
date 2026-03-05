\
# Streamlit app: Holton (2004) Ch. 12 — M12.2 style solutions with selectable diabatic forcing
#
# Run:
#   streamlit run holton_m12_2_streamlit_app.py
#
# This app computes (u, v*, w*, T) from the steady linear balances used in Holton Ch.12
# Problems 12.4–12.5 / MATLAB exercise M12.2, with user-selectable diabatic heating patterns.
#
# Notes:
# - To satisfy no-normal-flow at the side walls (v*=0 at y=0, Ly), we enforce that the meridional
#   integral of w* (and hence of the forcing) is zero at each z. For non-sinusoidal meridional
#   patterns we accomplish this by subtracting the meridional mean of the heating shape.

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st


# ----------------------------
# Helpers
# ----------------------------
def _cumulative_trapz_along_y(f_zy: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral in y, with integral = 0 at y[0]. f_zy shape (nz, ny)."""
    dy = np.diff(y)
    out = np.zeros_like(f_zy)
    areas = 0.5 * (f_zy[:, 1:] + f_zy[:, :-1]) * dy[None, :]
    out[:, 1:] = np.cumsum(areas, axis=1)
    return out


def _diverging_norm_if_needed(field: np.ndarray):
    """Return (cmap, norm) with a diverging norm centered at 0 when field spans +/-."""
    vmin = float(np.nanmin(field))
    vmax = float(np.nanmax(field))
    if vmin < 0.0 < vmax:
        vabs = max(abs(vmin), abs(vmax))
        norm = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)
        cmap = "RdBu_r"
        return cmap, norm
    return "viridis", None


def _contour_fig(y_km, z_km, field, title, units):
    cmap, norm = _diverging_norm_if_needed(field)
    fig, ax = plt.subplots()
    cf = ax.contourf(y_km, z_km, field, levels=21, cmap=cmap, norm=norm)
    ax.contour(y_km, z_km, field, levels=11, linewidths=0.5, colors="k", alpha=0.35)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(units)
    ax.set_xlabel("y (km)")
    ax.set_ylabel("z (km)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def build_forcing_shapes(
    y: np.ndarray,
    z: np.ndarray,
    forcing_kind: str,
    sigma_y_frac: float,
    sigma_z_frac: float,
    y_center_frac: float,
    z_center_frac: float,
):
    """
    Build separable shapes A(y)*B(z) used in w* = (J0/N^2) A(y) B(z).

    We always include sin(pi z/H) so that w*(z=0)=w*(z=H)=0.

    For non-sinusoidal A(y), subtract the meridional mean so that integral(A dy)=0
    (needed for v*=0 at y-walls when chi is defined by integrating w* in y).
    """
    Ly = y[-1] - y[0]
    H = z[-1] - z[0]

    if forcing_kind == "Holton (book): cos(ly)·sin(mz)":
        # With Ly=pi/l and H=pi/m, these reduce to cos(pi y/Ly) and sin(pi z/H)
        A = np.cos(math.pi * (y - y[0]) / Ly)
        B = np.sin(math.pi * (z - z[0]) / H)
        return A, B

    # Meridional Gaussian peak (high-lat or centered), then mean-subtract to enforce zero integral
    yc = y[0] + y_center_frac * Ly
    sigma_y = max(1e-9, sigma_y_frac * Ly)
    A = _gaussian(y, yc, sigma_y)

    # subtract meridional mean (continuous approx via trapezoid)
    A = A - (np.trapezoid(A, y) / Ly)

    # normalize so max |A| = 1
    A = A / max(1e-12, np.max(np.abs(A)))

    # Always enforce top/bottom impermeability
    B = np.sin(math.pi * (z - z[0]) / H)

    if forcing_kind == "Centered heating, shifted upward":
        zc = z[0] + z_center_frac * H
        sigma_z = max(1e-9, sigma_z_frac * H)
        B = B * _gaussian(z, zc, sigma_z)
        B = B / max(1e-12, np.max(np.abs(B)))

    return A, B


@st.cache_data(show_spinner=False)
def compute_fields(
    J0,
    N,
    f0,
    l,
    H,
    gamma,
    R,
    ny,
    nz,
    forcing_kind,
    sigma_y_frac,
    sigma_z_frac,
    y_center_frac,
    z_center_frac,
):
    """Compute w*, chi*, v*, u, T and J/cp for the chosen forcing."""
    Ly = math.pi / l
    y = np.linspace(0.0, Ly, ny)
    z = np.linspace(0.0, H, nz)

    A, B = build_forcing_shapes(
        y,
        z,
        forcing_kind,
        sigma_y_frac=sigma_y_frac,
        sigma_z_frac=sigma_z_frac,
        y_center_frac=y_center_frac,
        z_center_frac=z_center_frac,
    )

    # Separable w*: (J0/N^2) * A(y) * B(z)
    w = (J0 / (N**2)) * (B[:, None] * A[None, :])

    # Heating implied by thermo balance: (N^2 H / R) w* = J/cp
    Jcp = (N**2 * H / R) * w

    # Streamfunction chi* from w* = d(chi)/dy, with chi=0 at y=0
    chi = _cumulative_trapz_along_y(w, y)

    # v* = -d(chi)/dz
    v = -np.gradient(chi, z, axis=0, edge_order=2)

    # u from drag–Coriolis balance
    u = (f0 / gamma) * v

    # Thermal wind: f du/dz + (R/H) dT/dy = 0  -> dT/dy = -(H/R) f du/dz
    du_dz = np.gradient(u, z, axis=0, edge_order=2)
    dT_dy = -(H / R) * f0 * du_dz

    # integrate in y; choose T=0 at y=0 (additive constant arbitrary)
    T = _cumulative_trapz_along_y(dT_dy, y)

    return y, z, Jcp, w, v, u, T, chi


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Holton Ch.12 (M12.2) — Forcing-driven TEM fields", layout="wide")

st.title("Holton Ch.12 (M12.2) — Steady TEM response to prescribed diabatic forcing")
st.markdown(
    """
This app computes the steady fields \u03C7\\*, \(v^*\), \(w^*\), \(u\), and \(T\) from the simplified
balances used in Holton Ch.12 Problems 12.4–12.5 / Exercise M12.2, but lets you **switch the diabatic forcing pattern**.
"""
)

with st.sidebar:
    st.header("Diabatic forcing options")
    forcing_kind = st.selectbox(
        "Select forcing pattern",
        [
            "Holton (book): cos(ly)·sin(mz)",
            "High-latitude heating (Gaussian in y)",
            "Centered heating (Gaussian in y)",
            "Centered heating, shifted upward",
        ],
        index=0,
    )

    st.header("Model parameters (M12.2 defaults)")
    J0 = st.number_input("J0 (s⁻³)", value=1e-6, format="%.2e")
    N = st.number_input("N (s⁻¹)", value=1e-2, format="%.2e")
    f0 = st.number_input("f (s⁻¹)", value=1e-4, format="%.2e")
    l = st.number_input("l (m⁻¹)", value=1e-6, format="%.2e")
    H = st.number_input("H (m)", value=1e4, format="%.0f")
    gamma = st.number_input("γ (s⁻¹)", value=1e-5, format="%.2e")
    R = st.number_input("R (J kg⁻¹ K⁻¹)", value=287.0, format="%.1f")

    st.header("Grid")
    ny = st.slider("ny (meridional points)", 81, 401, 241, step=20)
    nz = st.slider("nz (vertical points)", 51, 301, 161, step=10)

    st.header("Gaussian-shape controls (non-book options)")
    if forcing_kind == "High-latitude heating (Gaussian in y)":
        y_center_frac_default = 0.85
    else:
        y_center_frac_default = 0.50

    y_center_frac = st.slider("Meridional center (fraction of channel width)", 0.0, 1.0, y_center_frac_default, 0.01)
    sigma_y_frac = st.slider("Meridional width σy (fraction of channel width)", 0.05, 0.50, 0.15, 0.01)

    if forcing_kind == "Centered heating, shifted upward":
        z_center_frac_default = 0.70
    else:
        z_center_frac_default = 0.50

    z_center_frac = st.slider("Vertical center (fraction of H)", 0.0, 1.0, z_center_frac_default, 0.01)
    sigma_z_frac = st.slider("Vertical width σz (fraction of H)", 0.05, 0.60, 0.20, 0.01)

y, z, Jcp, w, v, u, T, chi = compute_fields(
    J0,
    N,
    f0,
    l,
    H,
    gamma,
    R,
    ny,
    nz,
    forcing_kind,
    sigma_y_frac,
    sigma_z_frac,
    y_center_frac,
    z_center_frac,
)

# km axes
y_km = y / 1000.0
z_km = z / 1000.0

st.subheader("Prescribed diabatic forcing and resulting steady response")

colA, colB = st.columns(2)
with colA:
    st.pyplot(_contour_fig(y_km, z_km, Jcp, "Imposed diabatic heating J/cp", "K s$^{-1}$"), clear_figure=True)
with colB:
    st.pyplot(_contour_fig(y_km, z_km, w, "Residual vertical wind w*", "m s$^{-1}$"), clear_figure=True)

col1, col2 = st.columns(2)
with col1:
    st.pyplot(_contour_fig(y_km, z_km, v, "Residual meridional wind v*", "m s$^{-1}$"), clear_figure=True)
with col2:
    st.pyplot(_contour_fig(y_km, z_km, u, "Zonal-mean zonal wind u", "m s$^{-1}$"), clear_figure=True)

col3, col4 = st.columns(2)
with col3:
    st.pyplot(_contour_fig(y_km, z_km, T, "Temperature anomaly T (additive constant arbitrary)", "K (scaled)"), clear_figure=True)
with col4:
    st.pyplot(_contour_fig(y_km, z_km, chi, "Residual streamfunction χ*", "m$^2$ s$^{-1}$"), clear_figure=True)

st.markdown(
    """
**Colormaps:** If a field spans negative and positive values, the plot uses a diverging colormap
centered at zero (so 0 is visually neutral). Otherwise it falls back to a sequential colormap.
"""
)
