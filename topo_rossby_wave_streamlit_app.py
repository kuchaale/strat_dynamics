
"""
topo_rossby_wave_streamlit_app.py

Streamlit UI for the Holton Ch.12 topographic baroclinic Rossby-wave model.

Assumes these files are in the same folder:
  - topo_rossby_wave.py   (contains run_topo_rossby_wave)

Run:
  pip install streamlit numpy matplotlib
  streamlit run topo_rossby_wave_streamlit_app.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Make sure we can import topo_rossby_wave.py from the same folder as this app
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from topo_rossby_wave import run_topo_rossby_wave  # noqa: E402


st.set_page_config(page_title="Topographic baroclinic Rossby wave", layout="wide")
st.title("Topographic baroclinic Rossby wave – Holton Ch.12")


@st.cache_data(show_spinner=False)
def run_cached(**kwargs):
    return run_topo_rossby_wave(**kwargs)


def make_npz_bytes(out, meta: dict) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        deg=out.deg,
        zz_km=out.zz_km,
        hx_m=out.hx_m,
        psiz=out.psiz,
        xsiz=out.xsiz,
        theta=out.theta,
        **{f"meta_{k}": np.array(v) for k, v in meta.items()},
    )
    buf.seek(0)
    return buf.read()


with st.sidebar:
    st.header("Parameters")

    U = st.slider("Mean zonal wind U (m/s)", 0.0, 60.0, 20.0, 1.0)
    lat = st.slider("Latitude (deg)", 0.0, 80.0, 45.0, 1.0)
    hm_m = st.slider("Ridge height hₘ (m)", 0.0, 4000.0, 2000.0, 100.0)
    L_km = st.slider("Ridge half-width L (km)", 100.0, 3000.0, 800.0, 50.0)
    Lz_km = st.slider("Model top Lz (km)", 10.0, 60.0, 30.0, 1.0)
    r = st.selectbox("Linear damping r (s⁻¹)", options=[0.0, 1e-6, 2e-6, 5e-6], index=2)

    run_clicked = st.button("Run / Re-run", type="primary")

if "topo_out" not in st.session_state:
    st.session_state.topo_out = None
    st.session_state.topo_meta = None

meta = dict(U=float(U), lat=float(lat), hm_m=float(hm_m), L_km=float(L_km), Lz_km=float(Lz_km), r=float(r))

if run_clicked or st.session_state.topo_out is None:
    with st.spinner("Computing steady response…"):
        st.session_state.topo_out = run_cached(
            U=float(U),
            lat=float(lat),
            hm_m=float(hm_m),
            L_km=float(L_km),
            Lz_km=float(Lz_km),
            r=float(r),
            plot=False,
        )
        st.session_state.topo_meta = meta

out = st.session_state.topo_out

# geopotential height proxy: (f/g) psi
omega = 7.2921e-5
g = 9.81
latr = np.deg2rad(lat)
cor = 2.0 * omega * np.sin(latr)
geopot = (cor / g) * out.psiz

c1, c2, c3, c4 = st.columns(4)
c1.metric("U (m/s)", f"{U:.1f}")
c2.metric("lat (deg)", f"{lat:.1f}")
c3.metric("hₘ (m)", f"{hm_m:.0f}")
c4.metric("L (km)", f"{L_km:.0f}")

st.divider()

tab_xz, tab_slice, tab_download = st.tabs(["X–Z fields", "Ridge & height slice", "Download"])

with tab_xz:
    st.subheader("Longitude–height structure")

    colA, colB = st.columns(2)

    with colA:
        fig1, ax1 = plt.subplots()
        pcm = ax1.pcolormesh(out.deg, out.zz_km, geopot, shading="auto")
        fig1.colorbar(pcm, ax=ax1, label="m (proxy)")
        ax1.set_xlabel("longitude (deg)")
        ax1.set_ylabel("height (km)")
        ax1.set_title("Geopotential height (f/g · ψ)")
        st.pyplot(fig1, clear_figure=True)

    with colB:
        levels = st.multiselect(
            "θ contour levels (K)",
            options=np.arange(300,720,20),
            default=[300, 360, 400, 460, 600],
        )
        fig2, ax2 = plt.subplots()
        cs = ax2.contour(out.deg, out.zz_km, out.theta, levels=sorted(levels))
        ax2.clabel(cs, inline=True, fontsize=8)
        ax2.set_xlabel("longitude (deg)")
        ax2.set_ylabel("height (km)")
        ax2.set_title("Potential temperature proxy θ")
        st.pyplot(fig2, clear_figure=True)

with tab_slice:
    st.subheader("Topography and a single-height response")

    z_pick = st.slider("Pick height (km)", float(out.zz_km.min()), float(out.zz_km.max()), 4.5, 0.5)
    j = int(np.argmin(np.abs(out.zz_km - z_pick)))

    col1, col2 = st.columns(2)

    with col1:
        fig3, ax3 = plt.subplots()
        ax3.plot(out.deg, out.hx_m)
        ax3.set_xlabel("longitude (deg)")
        ax3.set_ylabel("ridge height (m)")
        ax3.set_title("Ridge profile h(x)")
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3, clear_figure=True)

    with col2:
        fig4, ax4 = plt.subplots()
        ax4.plot(out.deg, geopot[j, :])
        ax4.set_xlabel("longitude (deg)")
        ax4.set_ylabel("m (proxy)")
        ax4.set_title(f"Geopotential height at z ≈ {out.zz_km[j]:.1f} km")
        ax4.grid(True, alpha=0.3)
        st.pyplot(fig4, clear_figure=True)

with tab_download:
    st.subheader("Export results")
    npz_bytes = make_npz_bytes(out, meta)

    st.download_button(
        label="Download outputs (.npz)",
        data=npz_bytes,
        file_name=f"topo_rossby_U{U:.0f}_lat{lat:.0f}_hm{hm_m:.0f}_L{L_km:.0f}.npz",
        mime="application/octet-stream",
    )

    st.write("Preview shapes:")
    st.write(
        {
            "deg": out.deg.shape,
            "zz_km": out.zz_km.shape,
            "hx_m": out.hx_m.shape,
            "psiz": out.psiz.shape,
            "theta": out.theta.shape,
        }
    )
