#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy import fft


# %%
"""parameters for electron momentum simulation"""
m_e = 0.51099895000 * 10**(6) #eV/c2
m_e_err = 0.00000000015 * 10**(6) #eV/c2
detector_width = 6.35 * 10**(-4) #m
detector_length = 1.5 #m
fermi_E = 7.0 #eV
fermi_p = np.sqrt(2 * fermi_E * m_e) #eV/c
core_E_std = 4 + m_e #eV
core_p_std = np.sqrt(core_E_std**2 - m_e**2) #eV/c
c = 1 #c

# %%
class CoreElectron():
    def __init__(self, loc=0.0, scale=core_p_std):
        self.loc = loc
        self.scale = scale

    def get_pz(self, num):
        pz = np.random.normal(loc=self.loc, scale=self.scale, size=num)
        return pz

# %% Valence Electrons
class ValenceElectron():
    def __init__(self, fermi_p):
        self.fermi_p = fermi_p
        self.num_states = (4/3) * np.pi * self.fermi_p**3

    def get_p(self, num):
        r, theta, azi = self.gen_pos(num)
        px = r * np.cos(azi) * np.sin(theta)
        py = r * np.sin(azi) * np.sin(theta)
        pz = r * np.cos(theta)
        return(np.asarray[px, py, pz])

    def get_pz(self, num):
        r, theta, azi = self.gen_pos(num)
        pz = r*np.cos(theta)
        return pz

    def gen_pos(self, num):
        state = np.random.rand(num)*self.num_states
        r = np.cbrt(3 * state / (4 * np.pi))
        theta = np.random.rand(num) * np.pi
        azi = 2 * np.random.rand(num) * np.pi
        return r, azi, theta

# %%
class Interaction():
    def __init__(self, core, val, mass_e=m_e, l=detector_length, c=1):
        self.core = core
        self.val = val
        self.m_e = mass_e
        self.l = l
        self.c = c

    def get_E(self, pz):
        E = np.sqrt(pz**2 + 2 * self.m_e**2)
        return E

    def get_deflection(self, pz):
        E = self.get_E(pz)
        frac = self.c*pz/E
        deflection = np.arctan(frac)*2
        return deflection

    def get_z(self, pz):
        defl = self.get_deflection(pz)
        d = self.l * np.tan(defl)
        return d

    def predict_deflection(self, num):
        num_rand = np.random.rand(num) * 29
        num_v = (num_rand < 2.0).sum()
        num_c = num_detected - num_v
        pz_c = self.core.get_pz(num_c)
        pz_v = self.val.get_pz(num_v)
        d_v = self.get_z(pz_val)
        d_c = self.get_z(pz_core)
        return d_c, d_v





# %%
core = CoreElectron(scale=core_p_std)
val = ValenceElectron(fermi_p)
i = Interaction(m_e, detector_length)
num = 10**8
pz_val = val.get_pz(num)
pz_core = core.get_pz(num)
d_val = i.get_z(pz_val)
d_core = i.get_z(pz_core)

num_bins = 52
bins = (np.arange(num_bins) - num_bins // 2) * detector_width


hist_val = np.histogram(d_val, bins=bins)[0]
hist_core = np.histogram(d_core, bins=bins)[0]
hist_val = hist_val / np.sum(hist_val)
hist_core = hist_core / np.sum(hist_core)

fft_val = fft.fft(hist_val)
fft_core = fft.fft(hist_core)


# %%
num_bins = 51
bins = (np.arange(num_bins) - num_bins // 2) * detector_width

fig1, ax1 = plt.subplots()
ax1.bar(bins, hist_core, width=detector_width)
ax1.set_xlabel("z [m]")
ax1.set_ylabel("Population Density")
ax1.set_title("Core Electron Photon Deflection")
fig1.tight_layout()
fig1.show()

fig2, ax2 = plt.subplots()
ax2.bar(bins, hist_val, width=detector_width)
ax2.set_xlabel("z [m]")
ax2.set_ylabel("Population Density")
ax2.set_title("Valence Electron Photon Deflection")
fig2.tight_layout()
fig2.show()

fig3, ax3 = plt.subplots(2, 1)
ax3[0].bar(bins, hist_core, width=detector_width)
ax3[0].set_ylabel("Population Density")
ax3[0].set_title("Core Electron Photon Deflection")
ax3[1].bar(bins, hist_val, width=detector_width)
ax3[1].set_ylabel("Population Density")
ax3[1].set_xlabel("z [m]")
ax3[1].set_title("Valence Electron Photon Deflection")
fig3.tight_layout()
fig3.show()


# %%
"""Predicted distribution based on ratio of core electrons to valence electrons in copper"""
core = CoreElectron(scale=core_p_std)
val = ValenceElectron(fermi_p)
i = Interaction(core, val)
num_detected = 10**6
d_c, d_v = i.predict_deflection(num_detected)
d = np.concatenate((d_c, d_v))

num_bins = 52
bins = (np.arange(num_bins) - num_bins // 2) * detector_width
hist_d = np.histogram(d, bins=bins)[0]
hist_d = hist_d / np.sum(hist_d)

# %%
num_bins = 51
bins = (np.arange(num_bins) - num_bins // 2) * detector_width

fig4, ax4 = plt.subplots()
ax4.bar(bins, hist_d, width=detector_width)
ax4.set_xlabel("z [m]")
ax4.set_ylabel("Population Density")
ax4.set_title("Deflection of photon with respect to z for Cu")
fig4.tight_layout()
fig4.show()


# %%
data = np.genfromtxt('real_data.csv', delimiter=',', skip_header=1)
time = data[:, 1]
pos_int = data[:, 2]
counts = data[:, 3]
coincident_counts = data[:, 4]

norm_counts = coincident_counts / counts
norm_counts = norm_counts / np.sum(norm_counts)
pos = pos_int * detector_width

fig5, ax5 = plt.subplots()
ax5.bar(pos, coincident_counts, width=detector_width)
ax5.set_xlabel("z [m]")
ax5.set_ylabel("Population Density")
ax5.set_title("Observed deflection of photon with respect to z for Cu")
fig5.tight_layout()
fig5.show()

fig6, ax6 = plt.subplots()
ax6.bar(pos, norm_counts, width=detector_width)
ax6.set_xlabel("z [m]")
ax6.set_ylabel("Population Density")
ax6.set_title("Observed deflection of photon with respect to z for Cu")
fig6.tight_layout()
fig6.show()

# %%
fft_real = fft.fft(norm_counts)

fig7, ax7 = plt.subplots()
ax7.plot(fft_real, label='Real')
ax7.plot(fft_core, label='Core')
ax7.plot(fft_val, label='Valence')
ax7.set_xlabel("Coefficient")
ax7.set_ylabel("Population Density")
ax7.set_title("Observed deflection of photon with respect to z for Cu")
ax7.legend()
fig7.tight_layout()
fig7.show()


# %%
z = 28.0/29 * hist_core + 1.0/29 * hist_val
fig8, ax8 = plt.subplots()
ax8.plot(z, ':')
fig8.show()

print(a * 29)










