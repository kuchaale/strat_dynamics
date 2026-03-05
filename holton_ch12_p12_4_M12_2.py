\
# Holton (2004) Ch. 12 — Problem 12.4 and MATLAB Exercise M12.2 (Python)
# Analytic solution + contour plots for u, v*, w*, T (and J/cp)
#
# Run: python holton_ch12_p12_4_M12_2.py

import numpy as np
import matplotlib.pyplot as plt
import math

# --- M12.2 parameters ---
J0 = 1e-6      # s^-3
N  = 1e-2      # s^-1
f0 = 1e-4      # s^-1
l  = 1e-6      # m^-1
H  = 1e4       # m
m  = math.pi/H # m^-1
gamma = 1e-5   # s^-1
R = 287.0      # J kg^-1 K^-1 (dry air)

# --- Domain ---
ny, nz = 241, 161
y = np.linspace(0, math.pi/l, ny)
z = np.linspace(0, math.pi/m, nz)  # equals H
Y, Z = np.meshgrid(y, z, indexing="xy")

# --- Analytic solutions ---
chi = (J0/(l*N**2)) * np.sin(l*Y) * np.sin(m*Z)
w   = (J0/(N**2))   * np.cos(l*Y) * np.sin(m*Z)
v   = -(J0*m/(l*N**2)) * np.sin(l*Y) * np.cos(m*Z)
u   = (f0/gamma) * v
T   = (H/R) * (f0**2 * J0 * m**2)/(gamma * l**2 * N**2) * np.cos(l*Y) * np.sin(m*Z)
Jcp = (H/R) * J0 * np.cos(l*Y) * np.sin(m*Z)

# km axes for plotting
y_km = y/1000
z_km = z/1000

def contour_plot(field, title, units, fname):
    plt.figure()
    cf = plt.contourf(y_km, z_km, field, levels=21)
    plt.contour(y_km, z_km, field, levels=11, linewidths=0.5)
    plt.colorbar(cf, label=units)
    plt.xlabel("y (km)")
    plt.ylabel("z (km)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.show()

if __name__ == "__main__":
    contour_plot(u,   "Zonal-mean zonal wind u", "m s$^{-1}$", "u.png")
    contour_plot(v,   "Residual meridional wind v*", "m s$^{-1}$", "vstar.png")
    contour_plot(w,   "Residual vertical wind w*", "m s$^{-1}$", "wstar.png")
    contour_plot(T,   "Temperature anomaly T (additive constant arbitrary)", "K (scaled)", "T.png")
    contour_plot(Jcp, "Imposed diabatic heating J/cp", "K s$^{-1}$", "Jcp.png")
