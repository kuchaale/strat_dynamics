
"""
sudden_warming_model.py

Python/Numpy port of the MATLAB script:
  sudden_warming_model.m

Model: quasi-geostrophic channel model for wave–mean-flow interaction
(see comments in the original MATLAB file).

This port keeps the original finite-difference/AB3 time stepping, but
implements the (tri)diagonal linear algebra with fast tridiagonal
matvec/solves (Thomas algorithm), so it runs quickly without needing
SciPy.

Usage (CLI):
  python sudden_warming_model.py --s 2 --hb 200 --days 90 --plot

Or import:
  from sudden_warming_model import run_model
  out = run_model(s=2, hb=200, days=90)
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np


def sech(x: np.ndarray) -> np.ndarray:
    return 1.0 / np.cosh(x)


def tridiag_matvec(a: np.ndarray, b: np.ndarray, c: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Multiply a tridiagonal matrix by x.

    A has:
      subdiagonal a (len n-1)
      diagonal     b (len n)
      superdiag    c (len n-1)
    """
    y = b * x
    y[1:] += a * x[:-1]
    y[:-1] += c * x[1:]
    return y


def tridiag_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Solve Ax=d for tridiagonal A (Thomas algorithm).
    Works for real or complex arrays.

    a: subdiag (n-1)
    b: diag   (n)
    c: super  (n-1)
    d: rhs    (n)
    """
    n = b.size
    dtype = np.result_type(a, b, c, d)

    ac = np.array(a, dtype=dtype, copy=True)
    bc = np.array(b, dtype=dtype, copy=True)
    cc = np.array(c, dtype=dtype, copy=True)
    dc = np.array(d, dtype=dtype, copy=True)

    for i in range(1, n):
        m = ac[i - 1] / bc[i - 1]
        bc[i] = bc[i] - m * cc[i - 1]
        dc[i] = dc[i] - m * dc[i - 1]

    x = np.empty(n, dtype=dtype)
    x[-1] = dc[-1] / bc[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (dc[i] - cc[i] * x[i + 1]) / bc[i]
    return x


@dataclass
class ModelOutput:
    z_km: np.ndarray
    ub_initial: np.ndarray
    ub_final: np.ndarray
    ubtime: np.ndarray
    psitime: np.ndarray
    psi0_final: float
    psi_final: np.ndarray


def run_model(s: int = 2, hb: float = 200.0, days: int = 90, dt: float = 3600.0) -> ModelOutput:
    """
    Run the sudden warming model.

    Parameters
    ----------
    s : int
        Planetary (zonal) wavenumber.
    hb : float
        Lower boundary height disturbance (m).
    days : int
        Integration length (days). MATLAB suggests >= 90 days.
    dt : float
        Time step (s). MATLAB default is 3600 s.

    Returns
    -------
    ModelOutput
    """
    nl = 46
    ztop = 90e3
    nlm = nl - 1
    rad = 6.37e6

    k = s / (rad * np.cos(np.pi / 3.0))  # zonal wavenumber
    l = 3.0 / rad                        # meridional wavenumber

    days = int(days)
    dt = float(dt)
    dt12 = dt / 12.0
    steps_per_day = int(round(24.0 * 3600.0 / dt))
    ntime = days * steps_per_day

    dz = ztop / nlm
    dz2 = dz * dz

    z = np.linspace(dz, ztop, nlm)  # prediction levels (m)
    sh = 7000.0
    rho = np.exp(-z / sh)
    rho2 = np.sqrt(rho)

    # Damping profiles
    alph = (1.5 + np.tanh((z - 19000.0) / 7000.0)) * 1e-6
    dalph = 1e-6 * (sech((z - 19000.0) / 7000.0) ** 2) / 7000.0
    gamma = (1.0 + np.tanh((z - 40000.0) / 15000.0)) * 1e-6

    # Radiative equilibrium & initial mean wind
    ubrad0 = 0.0
    ubrad = ubrad0 + 2e-3 * z
    ub0 = 15.0
    ub = ub0 + 45.0 * np.cos(np.pi * (z * 1e-3 - 45.0) / 90.0)
    ubref = ub.copy()

    # Stability
    bv = 4e-4

    # Storage (daily)
    ubtime = np.zeros((nl, days))
    psitime = np.zeros((nl, days))

    # Parameters
    cor = 1.46e-4 * np.sin(np.pi / 3.0)
    beta = 1.46e-4 * np.cos(np.pi / 3.0) / rad
    cobv = cor * cor / bv
    dzlc = dz2 * l * l / cobv
    p = dalph / alph
    eps = 8.0 / (3.0 * np.pi)
    K1 = bv * (k * k + l * l) / (cor * cor)
    lkeps = l * l * k * eps / 2.0
    sig = 2.0 + dz2 / (4.0 * sh * sh) - p * dz2 / (2.0 * sh)
    tau = 2.5e5

    # Coeffs used in diagonals
    Mk = 1.0 + p * dz / 2.0
    Nk = 1.0 - p * dz / 2.0
    ujm = 1.0 + dz / (2.0 * sh)
    ujp = 1.0 - dz / (2.0 * sh)
    Mu = ujp + p * dz / 2.0
    Nu = ujm - p * dz / 2.0

    n = nlm

    # A1 for betae
    A1_sub = ujm * np.ones(n - 1)
    A1_diag = -2.0 * np.ones(n)
    A1_sup = ujp * np.ones(n - 1)
    A1_diag[-1] += ujp

    # Au for mean wind step
    Au_sub = ujm * np.ones(n - 1)
    Au_diag = -(2.0 + dzlc) * np.ones(n)
    Au_sup = ujp * np.ones(n - 1)
    Au_diag[-1] += ujp

    # Au1 for mean wind tendency
    Au1_sub = Nu[1:].copy()
    Au1_diag = -2.0 * np.ones(n)
    Au1_sup = Mu[:-1].copy()
    Au1_diag[-1] += Mu[-1]

    Au2 = np.zeros(n)
    Au2[0] = Nu[0] * (ub0 - ubrad0)
    Au2[-1] = -Mu[-1] * (ubrad[-1] - ubrad[-2])

    # Dupsi base (modified each step at top boundary)
    Dupsi_sub_base = np.ones(n - 1, dtype=np.complex128)
    Dupsi_diag_base = -2.0 * np.ones(n, dtype=np.complex128)
    Dupsi_sup_base = np.ones(n - 1, dtype=np.complex128)

    # AB3 storage
    psi = np.zeros(n, dtype=np.complex128)
    FKn = np.zeros(n, dtype=np.complex128)
    FKn1 = np.zeros(n, dtype=np.complex128)
    FKn2 = np.zeros(n, dtype=np.complex128)
    FAn = np.zeros(n, dtype=np.float64)
    FAn1 = np.zeros(n, dtype=np.float64)
    FAn2 = np.zeros(n, dtype=np.float64)

    # Wave matrix constants
    QK = 1.0 / (4.0 * sh * sh) + K1
    B = dz2 * QK

    tsec = 0.0
    psi0 = 0.0

    for istep in range(ntime):
        # betae
        A2 = np.zeros(n)
        A2[0] = ujm * ub0
        dub = tridiag_matvec(A1_sub, A1_diag, A1_sup, ub)
        betae = beta + eps * l * l * ub - cobv * eps / dz2 * (dub + A2)

        # radiation condition at the top
        m2 = (-QK + betae[-1] / (cobv * eps * ub[-1])) + 0j
        imdz = 1j * 2.0 * dz * np.sqrt(m2)

        # S1 diagonals
        S1_sub = Nk[1:].astype(np.complex128)
        S1_diag = (-sig).astype(np.complex128)
        S1_sup = Mk[:-1].astype(np.complex128)
        S1_diag[-1] = -sig[-1] + imdz * Mk[-1]
        S1_sub[-1] = 2.0

        # Dupsi diagonals
        Dupsi_sub = Dupsi_sub_base.copy()
        Dupsi_diag = Dupsi_diag_base.copy()
        Dupsi_sup = Dupsi_sup_base.copy()
        Dupsi_diag[-1] = -(2.0 + imdz)
        Dupsi_sub[-1] = 2.0

        # lower boundary forcing
        psi0m = 9.8 * hb / cor * (1.0 - np.exp(-(tsec + dt) / tau))
        psi0 = 9.8 * hb / cor * (1.0 - np.exp(-tsec / tau))

        S2 = np.zeros(n, dtype=np.complex128)
        S2[0] = Nk[0] * psi0

        # MA diagonals
        MA_sub = np.ones(n - 1, dtype=np.complex128)
        MA_diag = (-(2.0 + B)) * np.ones(n, dtype=np.complex128)
        MA_sup = np.ones(n - 1, dtype=np.complex128)
        MA_diag[-1] = (-(2.0 + B) + imdz)
        MA_sub[-1] = 2.0

        MA1 = np.zeros(n, dtype=np.complex128)
        MA2 = np.zeros(n, dtype=np.complex128)
        MA1[0] = psi0 - psi0m
        MA2[0] = psi0

        MApsi = tridiag_matvec(MA_sub, MA_diag, MA_sup, psi) + MA2
        S1psi = tridiag_matvec(S1_sub, S1_diag, S1_sup, psi) + S2

        FKn = (-1j * k * (eps * ub * MApsi + (dz2 / cobv) * betae * psi) - alph * S1psi)

        rhs_psi = MA1 + dt12 * (23.0 * FKn - 16.0 * FKn1 + 5.0 * FKn2)
        psi = psi + tridiag_solve(MA_sub, MA_diag, MA_sup, rhs_psi)

        FKn2, FKn1 = FKn1, FKn

        # EP flux convergence term
        Dupsi2 = np.zeros(n, dtype=np.complex128)
        Dupsi2[0] = 9.8 * hb / cor * (1.0 - np.exp(-tsec / tau))
        Dupsi_conjpsi = tridiag_matvec(Dupsi_sub, Dupsi_diag, Dupsi_sup, np.conjugate(psi)) + Dupsi2
        Cu = (lkeps / rho) * np.imag(psi * Dupsi_conjpsi)

        # mean wind tendency + step
        Au1_term = tridiag_matvec(Au1_sub, Au1_diag, Au1_sup, ub - ubrad) + Au2
        FAn = gamma * dzlc * ub - alph * Au1_term + Cu

        rhs_u = 23.0 * FAn - 16.0 * FAn1 + 5.0 * FAn2
        ub = ub + dt12 * tridiag_solve(Au_sub, Au_diag, Au_sup, rhs_u)

        FAn2, FAn1 = FAn1, FAn

        tsec += dt

        # save at daily intervals
        if (istep + 1) % steps_per_day == 0:
            day_idx = (istep + 1) // steps_per_day - 1
            ubtime[:, day_idx] = np.concatenate(([ub0], ub))
            psitime[:, day_idx] = np.concatenate(([psi0], np.abs(psi) / rho2)) * cor / 9.8

    z_km = np.concatenate(([16.0], 16.0 + z * 1e-3))
    ub_initial = np.concatenate(([ub0], ubref))
    ub_final = np.concatenate(([ub0], ub))

    return ModelOutput(
        z_km=z_km,
        ub_initial=ub_initial,
        ub_final=ub_final,
        ubtime=ubtime,
        psitime=psitime,
        psi0_final=float(psi0),
        psi_final=psi,
    )


def _plot(out: ModelOutput) -> None:
    import matplotlib.pyplot as plt

    # Final mean wind profile
    plt.figure()
    plt.plot(out.ub_final, out.z_km)
    plt.plot(out.ub_initial, out.z_km, "--")
    plt.xlabel("m/s")
    plt.ylabel("height (km)")
    plt.title("initial ubar (dashed); final ubar (solid)")
    plt.show()
    plt.savefig('profiles.png', bbox_inches = 'tight')

    # Time-height sections
    days = np.arange(1, out.ubtime.shape[1] + 1)
    D, Z = np.meshgrid(days, out.z_km)

    plt.figure()
    plt.pcolormesh(D, Z, out.ubtime, shading="auto")
    plt.colorbar(label="m/s")
    plt.xlabel("time (days)")
    plt.ylabel("height (km)")
    plt.title("mean zonal wind")
    #plt.show()
    plt.savefig('zw.png', bbox_inches = 'tight')

    plt.figure()
    plt.pcolormesh(D, Z, out.psitime, shading="auto")
    plt.colorbar(label="m")
    plt.xlabel("time (days)")
    plt.ylabel("height (km)")
    plt.title("wave geopotential height")
    #plt.show()
    plt.savefig('gh.png', bbox_inches = 'tight')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=int, default=2, help="planetary wavenumber (1,2,3,...)")
    ap.add_argument("--hb", type=float, default=200.0, help="lower boundary height disturbance (m)")
    ap.add_argument("--days", type=int, default=90, help="integration length (days)")
    ap.add_argument("--dt", type=float, default=3600.0, help="time step (s)")
    ap.add_argument("--plot", action="store_true", help="make matplotlib plots")
    args = ap.parse_args()

    t0 = time.time()
    out = run_model(s=args.s, hb=args.hb, days=args.days, dt=args.dt)
    t1 = time.time()

    print(f"Ran {args.days} days with dt={args.dt:g} s in {t1-t0:.3f} s.")
    print(f"Final ubar at ~16 km: {out.ub_final[0]:.3f} m/s")
    print(f"Final ubar at top:    {out.ub_final[-1]:.3f} m/s")

    if args.plot:
        _plot(out)


if __name__ == "__main__":
    main()
