
"""
qbo_model.py

Python/Numpy port of:
  qbo_model.m  (Holton, An Introduction to Dynamic Meteorology, Ch. 12)

Model:
  1-D analogue for the tropical QBO, based on Plumb (1977, J. Atmos. Sci.).
  Uses 3rd-order Adams–Bashforth (AB3) time stepping.
  Forcing is specified via lower-boundary momentum flux from two gravity waves
  of equal and opposite flux and phase speed.

This version:
  - Uses numpy only (matplotlib optional for plots).
  - Vectorizes the expensive inner j-loop from MATLAB using cumulative sums.
  - Saves the mean wind profile every 2 nondimensional time units.

Run:
  python qbo_model.py --Am 0.2 --time_end 200 --plot

Or import:
  from qbo_model import run_qbo_model
  out = run_qbo_model(Am=0.2, time_end=200)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass
class QBOOutput:
    zplot: np.ndarray        # (J+1,) height grid including 0
    t_save: np.ndarray       # saved times (every 2 units), (nsave,)
    ubtime: np.ndarray       # mean wind time-height, shape (J+1, nsave)
    ub_final: np.ndarray     # final profile, (J+1,)


def run_qbo_model(
    Am: float = 0.2,
    time_end: float = 200.0,
    dt: float = 0.01,
    J: int = 80,
    ztop: float = 4.0,
    k1: float = 1.0,
    k2: float = 1.0,
    c1: float = 1.0,
    c2: float = -1.0,
    alph: float = 1.0,
    lam: float = 0.02,
    plot: bool = False,
) -> QBOOutput:
    """
    Parameters mirror the MATLAB script where possible.

    Am : float
        Amplitude of boundary forcing (MATLAB suggests 0.04–0.4).
    time_end : float
        Integration time in nondimensional units (MATLAB suggests >= 100).
    dt : float
        Time step (default 0.01).
    J : int
        Vertical grid points (default 80).
    plot : bool
        Make matplotlib plots similar to MATLAB.

    Returns
    -------
    QBOOutput
    """
    J = int(J)
    dz = ztop / J
    dz2 = dz * dz
    Jl = J - 1

    # Prediction levels (MATLAB: z = linspace(dz,ztop,J))
    z = np.linspace(dz, ztop, J)
    zplot = np.concatenate(([0.0], z))

    # Save every 2 time units (MATLAB: save_t = fix(2/dt))
    save_t = int(np.floor(2.0 / dt))
    dt12 = dt / 12.0

    # Initial mean wind
    ub0 = 0.0
    ub = 0.2 * np.sin(np.pi * z / 4.0)  # size J

    # AB3 storage
    Force = np.zeros(J)
    Force1 = np.zeros(J)
    Force2 = np.zeros(J)

    # Output storage
    nsave_target = int(np.floor(time_end / 2.0))
    ubtime = np.zeros((J + 1, nsave_target))
    t_save = np.zeros(nsave_target)

    ntime = int(np.floor(time_end / dt))

    count = 0
    nsave = 0
    t = 0.0

    for _ in range(ntime):
        # Avoid division-by-zero if ub approaches wave phase speed exactly
        # (MATLAB would blow up too; this just prevents hard crashes)
        denom1 = k1 * (ub - c1) ** 2
        denom2 = k2 * (ub - c2) ** 2
        denom1 = np.where(denom1 == 0.0, np.finfo(float).tiny, denom1)
        denom2 = np.where(denom2 == 0.0, np.finfo(float).tiny, denom2)

        G1n = alph / denom1
        G2n = alph / denom2

        G10 = alph / (k1 * (ub0 - c1) ** 2)
        G20 = alph / (k2 * (ub0 - c2) ** 2)

        # MATLAB inner loop:
        #   exp(-(0.5*(G10+G1n(j)) + sum(G1n(1:j-1))) * dz)
        # vectorized via prefix sums
        prefix1 = np.concatenate(([0.0], np.cumsum(G1n[:-1])))
        prefix2 = np.concatenate(([0.0], np.cumsum(G2n[:-1])))

        F1n = Am * G1n * np.exp(-(0.5 * (G10 + G1n) + prefix1) * dz)
        F2n = -Am * G2n * np.exp(-(0.5 * (G20 + G2n) + prefix2) * dz)

        # Diffusion (second derivative)
        Force[:] = 0.0
        # Force(2:Jl) -> python indices 1 .. Jl-1 (inclusive) i.e. 1:Jl
        Force[1:Jl] = (
            F1n[1:Jl] + F2n[1:Jl]
            + lam / dz2 * (ub[2:J] - 2.0 * ub[1:J-1] + ub[0:J-2])
        )
        # Force(1) -> python index 0
        Force[0] = F1n[0] + F2n[0] + lam / dz2 * (ub[1] - 2.0 * ub[0] + ub0)

        # AB3 step for ub (MATLAB: ubm(1:Jl)=... ; ubm(J)=ubm(Jl))
        ubm = ub.copy()
        rhs = 23.0 * Force[:Jl] - 16.0 * Force1[:Jl] + 5.0 * Force2[:Jl]
        ubm[:Jl] = ub[:Jl] + dt12 * rhs
        ubm[J-1] = ubm[Jl-1]  # top boundary: ub(J)=ub(Jl)

        ub = ubm

        # Rotate AB3 storages
        Force2, Force1 = Force1, Force.copy()

        # time bookkeeping + save
        t += dt
        count += 1
        if count == save_t:
            if nsave < nsave_target:
                ubplot = np.concatenate(([ub0], ub))
                ubtime[:, nsave] = ubplot
                t_save[nsave] = t
            nsave += 1
            count = 0

    # Trim if time_end/dt didn't land exactly
    nsave_used = min(nsave, nsave_target)
    ubtime = ubtime[:, :nsave_used]
    t_save = t_save[:nsave_used]

    out = QBOOutput(
        zplot=zplot,
        t_save=t_save,
        ubtime=ubtime,
        ub_final=ubtime[:, -1] if ubtime.size else np.concatenate(([ub0], ub)),
    )

    if plot:
        _plot(out)

    return out


def _plot(out: QBOOutput) -> None:
    import matplotlib.pyplot as plt

    zplot = out.zplot
    t = out.t_save
    ubtime = out.ubtime

    # Figure 1: last saved profile (similar vibe to MATLAB)
    plt.figure()
    plt.plot(out.ub_final, zplot, "k")
    plt.axis([-1, 1, 0, float(np.max(zplot))])
    plt.xlabel("zonal wind (nondimensional)")
    plt.ylabel("height (nondimensional)")
    plt.title("mean zonal wind (final saved)")

    # Figure 2: time series at two heights (MATLAB uses indices 10 and 40)
    if ubtime.shape[1] > 0:
        plt.figure()
        i1 = 9   # MATLAB 10
        i2 = 39  # MATLAB 40
        i1 = min(i1, ubtime.shape[0]-1)
        i2 = min(i2, ubtime.shape[0]-1)
        plt.plot(t, ubtime[i1, :], label=f"z index {i1+1}")
        plt.plot(t, ubtime[i2, :], "--k", label=f"z index {i2+1}")
        plt.xlabel("time (nondimensional)")
        plt.ylabel("velocity (nondimensional)")
        plt.axis([0, float(t[-1]) if len(t) else 1.0, -1, 1])
        plt.title("mean zonal wind at two heights")
        plt.legend()

    # Figure 3: time-height section (pcolor + contours)
    if ubtime.shape[1] > 0:
        T, Z = np.meshgrid(t, zplot)
        plt.figure()
        plt.pcolormesh(T, Z, ubtime, shading="auto")
        plt.colorbar(label="u (nondim)")
        plt.xlabel("time")
        plt.ylabel("height (nondimensional)")
        plt.title("mean zonal wind (nondimensional)")
        plt.contour(T, Z, ubtime, levels=[-.8, -.6, -.4, -.2, 0, .2, .4, .6, .8], colors="k", linewidths=0.5)

    plt.show()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--Am", type=float, default=0.2, help="forcing amplitude (0.04–0.4 typical)")
    ap.add_argument("--time_end", type=float, default=200.0, help="integration length (nondimensional time units)")
    ap.add_argument("--dt", type=float, default=0.01, help="time step")
    ap.add_argument("--plot", action="store_true", help="show matplotlib figures")
    args = ap.parse_args()

    out = run_qbo_model(Am=args.Am, time_end=args.time_end, dt=args.dt, plot=args.plot)
    print(f"Saved ubtime array with shape {out.ubtime.shape}; saved times from {out.t_save[0] if len(out.t_save) else 'n/a'} to {out.t_save[-1] if len(out.t_save) else 'n/a'}")


if __name__ == "__main__":
    main()
