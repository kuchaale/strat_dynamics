
"""
ssw_streamlit_app.py

Streamlit UI for the Holton Ch.12 "sudden_warming_model" Python port.

Put this file next to:
  - sudden_warming_model.py   (contains run_model)

Run:
  pip install streamlit numpy matplotlib
  streamlit run ssw_streamlit_app.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Make sure we can import sudden_warming_model.py from the same folder as this app
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from sudden_warming_model_epf import run_model  # noqa: E402


st.set_page_config(page_title="Holton SSW toy model", layout="wide")


@st.cache_data(show_spinner=False)
def run_cached(s: int, hb: float, days: int, dt: float):
    """
    Cache model runs so normal Streamlit reruns (slider moves, tab switches)
    don't recompute the simulation each time.
    """
    return run_model(s=s, hb=hb, days=days, dt=dt)


def make_npz_bytes(out) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        z_km=out.z_km,
        ub_initial=out.ub_initial,
        ub_final=out.ub_final,
        ubtime=out.ubtime,
        psitime=out.psitime,
        epdivtime=out.epdivtime,
    )
    buf.seek(0)
    return buf.read()


st.title("Sudden Stratospheric Warming (SSW) toy model – Holton Ch.12")

with st.sidebar:
    st.header("Model controls")

    s = st.selectbox("Planetary wavenumber s", options=[1, 2, 3, 4], index=1)
    hb = st.slider("Lower-boundary forcing hb (m)", 0.0, 600.0, 200.0, 10.0)
    days = st.slider("Integration length (days)", 10, 180, 90, 5)
    dt = st.selectbox("Time step dt (s)", options=[1800.0, 3600.0, 7200.0], index=1)

    run_clicked = st.button("Run / Re-run model", type="primary")

# Persist results across reruns
if "out" not in st.session_state:
    st.session_state.out = None

if run_clicked or st.session_state.out is None:
    with st.spinner("Running model…"):
        st.session_state.out = run_cached(int(s), float(hb), int(days), float(dt))

out = st.session_state.out

# ---- Summary row
c1, c2, c3, c4 = st.columns(4)
c1.metric("s", int(s))
c2.metric("hb (m)", f"{hb:.0f}")
c3.metric("days", int(days))
c4.metric("dt (s)", f"{dt:.0f}")

st.divider()

# ---- Tabs
tab_profiles, tab_timeheight, tab_download = st.tabs(
    ["Profiles", "Time–height sections", "Download"]
)

with tab_profiles:
    st.subheader("Mean wind profiles")

    day_pick = st.slider(
        "Pick a day to plot ū(z, day)",
        min_value=1,
        max_value=int(days),
        value=min(30, int(days)),
        step=1,
    )
    u_day = out.ubtime[:, day_pick - 1]

    fig, ax = plt.subplots()
    ax.plot(out.ub_initial, out.z_km, linestyle="--", label="initial")
    ax.plot(out.ub_final, out.z_km, label="final")
    ax.plot(u_day, out.z_km, label=f"day {day_pick}")
    ax.axvline(0, ls = 'dashed', c = 'gray')
    ax.set_xlabel("ū (m/s)")
    ax.set_ylabel("height (km)")
    ax.set_title("Mean zonal wind profiles")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    st.caption("Dashed = initial; solid = final; thin line = selected day profile.")

with tab_timeheight:
    st.subheader("Time–height evolution")

    t_days = np.arange(1, out.ubtime.shape[1] + 1)

    col_u, col_psi, col_ep = st.columns(3)
    height = out.z_km
    mask = height <= 100

    with col_u:
        fig1, ax1 = plt.subplots()
        pcm = ax1.pcolormesh(t_days, height[mask], out.ubtime[mask], shading="auto", cmap = 'RdBu', vmin = -80, vmax = 80)
        fig1.colorbar(pcm, ax=ax1, label="m/s")
        ax1.set_xlabel("time (days)")
        ax1.set_ylabel("height (km)")
        ax1.set_title("Mean zonal wind ū")
        st.pyplot(fig1, clear_figure=True)

    with col_psi:
        fig2, ax2 = plt.subplots()
        pcm2 = ax2.pcolormesh(t_days, height[mask], out.psitime[mask], shading="auto")
        fig2.colorbar(pcm2, ax=ax2, label="m")
        ax2.set_xlabel("time (days)")
        ax2.set_ylabel("height (km)")
        ax2.set_title("Wave geopotential height (|ψ| scaled)")
        st.pyplot(fig2, clear_figure=True)

    with col_ep:
        fig3, ax3 = plt.subplots()

        calc_data = out.epdivtime[mask]*24*3600 # convert to m/s/day
        ROBUST_PERCENTILE = 10
        vmin = np.percentile(calc_data, ROBUST_PERCENTILE)
        vmax = np.percentile(calc_data, 100-ROBUST_PERCENTILE)
        center = 0
        vlim = max(abs(vmin - center), abs(vmax - center))
        vmin, vmax = -vlim, vlim
        pcm3 = ax3.pcolormesh(t_days, height[mask], calc_data, shading="auto", vmin = vmin, vmax = vmax, cmap = 'berlin')
        fig3.colorbar(pcm3, ax=ax3, label="m/s/day")
        ax3.set_xlabel("time (days)")
        ax3.set_ylabel("height (km)")
        ax3.set_title("EP flux divergence (∂F/∂z)")
        st.pyplot(fig3, clear_figure=True)

with tab_download:
    st.subheader("Export results")

    st.write("Download a compressed NumPy archive (`.npz`) containing the main outputs:")
    npz_bytes = make_npz_bytes(out)

    st.download_button(
        label="Download SSW outputs (.npz)",
        data=npz_bytes,
        file_name=f"ssw_s{s}_hb{int(hb)}_days{int(days)}_dt{int(dt)}.npz",
        mime="application/octet-stream",
    )

    st.write("Preview: final profile table (height, initial ū, final ū)")
    preview = np.column_stack([out.z_km, out.ub_initial, out.ub_final])
    st.dataframe(preview, use_container_width=True)
