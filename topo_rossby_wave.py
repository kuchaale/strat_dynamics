
"""
topo_rossby_wave.py

Python/Numpy port of:
  topo_Rossby_wave.m  (Holton, An Introduction to Dynamic Meteorology, Ch. 12)

Model:
  Steady, linear topographic baroclinic Rossby wave in a 1-D QG model.
  Ridge profile: h(x) = hm / (1 + ((x-xm)/L)^2)  (same as the MATLAB file).

This version:
  - Uses numpy.fft for the Fourier transform / inversion.
  - Keeps the MATLAB algebra and indexing choices as closely as possible.
  - Optional matplotlib plots.

Run:
  python topo_rossby_wave.py --U 20 --plot

Or import:
  from topo_rossby_wave import run_topo_rossby_wave
  out = run_topo_rossby_wave(U=20, plot=False)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TopoRossbyOutput:
    deg: np.ndarray          # longitude grid (degrees), length N
    zz_km: np.ndarray        # height grid (km), length NL
    hx_m: np.ndarray         # ridge profile (m), length N
    psiz: np.ndarray         # streamfunction in x,z (same units as MATLAB), shape (NL,N)
    xsiz: np.ndarray         # vertical displacement (m), shape (NL,N)
    theta: np.ndarray        # potential temperature proxy, shape (NL,N)


def run_topo_rossby_wave(
    U: float = 20.0,
    lat: float = 45.0,
    Lz_km: float = 30.0,
    L_km: float = 800.0,
    hm_m: float = 2000.0,
    Ly_km: float = 8000.0,
    H_m: float = 8000.0,
    r: float = 2.0e-6,
    NL: int = 61,
    N: int = 128,
    plot: bool = False,
) -> TopoRossbyOutput:
    """
    Parameters mirror the MATLAB script where possible.

    U : float
        Mean zonal wind (m/s). In MATLAB this is requested via input().
    lat : float
        Channel latitude (degrees).
    Lz_km : float
        Depth scale (km).
    L_km : float
        Zonal ridge half-width scale (km).
    hm_m : float
        Ridge height scale (m).
    r : float
        Linear damping rate (s^-1).
    NL, N : int
        Vertical gridpoints and Fourier modes.

    Returns
    -------
    TopoRossbyOutput
    """
    # Constants / parameters
    omega = 7.2921e-5
    rad = 6.37e6
    g = 9.81
    bv = 4.0e-4

    latr = np.pi * lat / 180.0
    cor = 2.0 * omega * np.sin(latr)
    beta = 2.0 * omega * np.cos(latr) / rad

    # Domain geometry (MATLAB uses km for x-domain lengths)
    Lx_km = 2.0 * np.pi * 6.37e3 * np.cos(latr)  # circumference at latitude (km)
    k = 2.0 * np.pi / (Lx_km * 1e3)              # lowest zonal wavenumber (1/m)
    m = np.pi / (8.0e6)                          # meridional wavenumber (1/m)

    # Grids
    zz_km = np.linspace(0.0, Lz_km, NL)
    xx_km = np.linspace(0.0, Lx_km, N)
    deg = np.linspace(0.0, 360.0, N)
    xm_km = Lx_km / 4.0

    # Ridge profile (MATLAB uses +hm/(1+((x-xm)/L)^2); comment says -hm but code is +)
    hx_m = hm_m / (1.0 + ((xx_km - xm_km) / L_km) ** 2)

    # Fourier transform of topography
    hn = np.fft.fft(hx_m, n=N)
    hn[0] = 0.0  # remove mean component (MATLAB: hn(1)=0)

    # Allocate spectral arrays
    psinz = np.zeros((NL, N), dtype=np.complex128)
    xsi = np.zeros((NL, N), dtype=np.complex128)

    s = np.arange(1, N + 1)  # MATLAB: s=[1:N] (not strictly needed)

    for n in range(2, N + 1):  # MATLAB: for n=2:N
        ns = n - 1
        Kn2 = (k * ns) ** 2 + m ** 2
        eps = r / (k * ns * U)  # damping parameter

        # Vertical wavenumber from analytic solution (keep MATLAB expression)
        Bn = (bv / cor**2) * (
            beta / (U * (1.0 + eps**2))
            - (Kn2 + cor**2 / (bv * 4.0 * H_m**2))
            + 1j * beta * eps / (U * (1.0 + eps**2))
        )

        sqrtBn = np.sqrt(Bn + 0j)  # enforce complex sqrt like MATLAB

        psin = -(bv / cor) * hn[n - 1] / ((1j * sqrtBn + 1.0 / (2.0 * H_m)) * (1.0 - 1j * eps))

        phase = 1j * sqrtBn * (zz_km * 1e3) + (zz_km * 1e3) / (2.0 * H_m)
        prof = np.exp(phase)

        psinz[:, n - 1] = psin * prof
        xsi[:, n - 1] = hn[n - 1] * prof / (1.0 - 1j * eps)

    # Inverse FFT to physical space for each height
    psiz = np.real(np.fft.ifft(psinz, n=N, axis=1))
    xsiz = np.real(np.fft.ifft(xsi, n=N, axis=1))

    # Potential temperature "trajectories" proxy from MATLAB
    theta = 300.0 * np.exp(bv / g * (zz_km[:, None] * 1e3)) - 300.0 * bv / g * xsiz

    out = TopoRossbyOutput(deg=deg, zz_km=zz_km, hx_m=hx_m, psiz=psiz, xsiz=xsiz, theta=theta)

    if plot:
        _plot(out, U=U, cor=cor, g=g, m=m, Ly_km=Ly_km, H_m=H_m)

    return out


def _plot(out: TopoRossbyOutput, U: float, cor: float, g: float, m: float, Ly_km: float, H_m: float) -> None:
    import matplotlib.pyplot as plt

    deg = out.deg
    zz = out.zz_km
    psiz = out.psiz
    theta = out.theta
    hx = out.hx_m

    x, z = np.meshgrid(deg, zz)

    # Figure 1: geopotential height proxy (MATLAB: cor/g*psiz)
    plt.figure()
    plt.pcolormesh(x, z, (cor / g) * psiz, shading="auto")
    plt.colorbar(label="m (proxy)")
    plt.title("geopotential height (proxy)")
    plt.xlabel("longitude (degrees)")
    plt.ylabel("height (km)")

    # Figure 2: theta contours (MATLAB uses specific contour levels)
    plt.figure()
    cs = plt.contour(x, z, theta, levels=[300, 320, 340, 360, 380, 400])
    plt.clabel(cs, inline=True, fontsize=8)
    plt.axis([0, 360, 0, 8])
    plt.title("potential temperature (proxy)")
    plt.xlabel("longitude (degrees)")
    plt.ylabel("height (km)")

    # Figure 3: ridge and geopotential height at z*=4.5 km (MATLAB uses psiz(10,:))
    j10 = 9  # MATLAB j=10 -> Python index 9
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(deg, hx)
    plt.axis([0, 360, 0, float(np.max(hx))])
    plt.ylabel("height (m)")
    plt.title("ridge profile")

    plt.subplot(2, 1, 2)
    geo_45 = np.real(psiz[j10, :]) * cor / 9.8
    plt.plot(deg, geo_45)
    plt.axis([0, 360, float(np.min(geo_45)), float(np.max(geo_45))])
    plt.ylabel("height (m)")
    plt.xlabel("longitude (degrees)")
    plt.title("geopotential height at z*=4.5 km")

    # Figure 4: y-longitude contours
    yy = np.linspace(0.0, Ly_km, 15)
    x2, y2 = np.meshgrid(deg, yy)

    psir, _ = np.meshgrid(psiz[j10, :], yy)
    hxy, _ = np.meshgrid(hx, yy)

    hxy_contour = hxy * np.sin(m * y2 * 1e3)
    psi_contour = psir * np.sin(m * y2 * 1e3) - U * y2 * 1e3

    plt.figure()
    plt.contour(x2, y2, hxy_contour, levels=[1000], linestyles="--")
    plt.ylabel("y (km)")
    plt.xlabel("longitude (degrees)")
    plt.contour(x2, y2, psi_contour)
    plt.title("1 km topography (dashed); streamfunction at z*=4.5 km (solid)")

    plt.show()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--U", type=float, default=20.0, help="mean zonal wind (m/s)")
    ap.add_argument("--plot", action="store_true", help="show matplotlib figures")
    args = ap.parse_args()

    out = run_topo_rossby_wave(U=args.U, plot=args.plot)
    print(f"Computed topo Rossby-wave response on grid: psiz shape = {out.psiz.shape}, theta shape = {out.theta.shape}")


if __name__ == "__main__":
    main()
