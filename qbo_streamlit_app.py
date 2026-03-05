
"""
qbo_streamlit_app.py

Streamlit UI for the Holton Ch.12 QBO toy model.

Assumes these files are in the same folder:
  - qbo_model.py   (contains run_qbo_model)

Run:
  pip install streamlit numpy matplotlib
  streamlit run qbo_streamlit_app.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Make sure we can import qbo_model.py from the same folder as this app
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from qbo_model import run_qbo_model  # noqa: E402


st.set_page_config(page_title="QBO toy model", layout="wide")
st.title("QBO toy model – Holton Ch.12")


@st.cache_data(show_spinner=False)
def run_cached(**kwargs):
    return run_qbo_model(**kwargs)


def make_npz_bytes(out, meta: dict) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        zplot=out.zplot,
        t_save=out.t_save,
        ubtime=out.ubtime,
        ub_final=out.ub_final,
        **{f"meta_{k}": np.array(v) for k, v in meta.items()},
    )
    buf.seek(0)
    return buf.read()


with st.sidebar:
    st.header("Parameters")

    Am = st.slider("Wave forcing amplitude Aₘ", 0.01, 0.6, 0.2, 0.01)
    time_end = st.slider("Integration length (time units)", 20.0, 600.0, 200.0, 10.0)
    dt = st.selectbox("Time step dt", options=[0.005, 0.01, 0.02], index=1)

    lam = st.slider("Diffusion λ", 0.0, 0.2, 0.02, 0.005)
    J = st.selectbox("Vertical grid points J", options=[60, 80, 100, 120], index=1)

    run_clicked = st.button("Run / Re-run", type="primary")

if "qbo_out" not in st.session_state:
    st.session_state.qbo_out = None
    st.session_state.qbo_meta = None

meta = dict(Am=float(Am), time_end=float(time_end), dt=float(dt), lam=float(lam), J=int(J))

if run_clicked or st.session_state.qbo_out is None:
    with st.spinner("Integrating QBO model…"):
        st.session_state.qbo_out = run_cached(
            Am=float(Am),
            time_end=float(time_end),
            dt=float(dt),
            lam=float(lam),
            J=int(J),
            plot=False,
        )
        st.session_state.qbo_meta = meta

out = st.session_state.qbo_out

c1, c2, c3, c4 = st.columns(4)
c1.metric("Aₘ", f"{Am:.2f}")
c2.metric("time_end", f"{time_end:.0f}")
c3.metric("dt", f"{dt:.3f}")
c4.metric("λ", f"{lam:.3f}")

st.divider()

tab_timeheight, tab_profiles, tab_download = st.tabs(["Time–height", "Profiles & time series", "Download"])

with tab_timeheight:
    st.subheader("Time–height evolution of mean wind")

    if out.ubtime.shape[1] == 0:
        st.warning("No saved output times (try increasing time_end or reducing dt).")
    else:
        fig, ax = plt.subplots()
        pcm = ax.pcolormesh(out.t_save, out.zplot, out.ubtime, shading="auto", cmap='Spectral_r', vmin = -1, vmax = 1)
        fig.colorbar(pcm, ax=ax, label="u (nondim)")
        ax.set_xlabel("time")
        ax.set_ylabel("height (nondim)")
        ax.set_title("Mean zonal wind ū(t,z)")
        st.pyplot(fig, clear_figure=True)

with tab_profiles:
    st.subheader("Final profile and selected-height time series")

    colA, colB = st.columns(2)

    with colA:
        fig1, ax1 = plt.subplots()
        ax1.plot(out.ub_final, out.zplot, "k")
        ax1.set_xlabel("u (nondim)")
        ax1.set_ylabel("height (nondim)")
        ax1.set_title("Final saved mean wind profile")
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1, clear_figure=True)

    with colB:
        if out.ubtime.shape[1] == 0:
            st.info("No saved time series to plot.")
        else:
            z1 = st.slider("Height #1 (nondim)", float(out.zplot.min()), float(out.zplot.max()), float(out.zplot[10]), 0.05)
            z2 = st.slider("Height #2 (nondim)", float(out.zplot.min()), float(out.zplot.max()), float(out.zplot[40 if len(out.zplot)>40 else -1]), 0.05)
            i1 = int(np.argmin(np.abs(out.zplot - z1)))
            i2 = int(np.argmin(np.abs(out.zplot - z2)))

            fig2, ax2 = plt.subplots()
            ax2.plot(out.t_save, out.ubtime[i1, :], label=f"z≈{out.zplot[i1]:.2f}")
            ax2.plot(out.t_save, out.ubtime[i2, :], linestyle="--", label=f"z≈{out.zplot[i2]:.2f}")
            ax2.set_xlabel("time")
            ax2.set_ylabel("u (nondim)")
            ax2.set_title("Mean wind at two heights")
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            st.pyplot(fig2, clear_figure=True)

with tab_download:
    st.subheader("Export results")
    npz_bytes = make_npz_bytes(out, meta)

    st.download_button(
        label="Download outputs (.npz)",
        data=npz_bytes,
        file_name=f"qbo_Am{Am:.2f}_T{time_end:.0f}_dt{dt:.3f}_J{J}.npz",
        mime="application/octet-stream",
    )

    st.write("Preview shapes:")
    st.write({"zplot": out.zplot.shape, "t_save": out.t_save.shape, "ubtime": out.ubtime.shape})
