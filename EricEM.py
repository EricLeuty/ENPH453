#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Eric Leuty
# %%
import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy.fft as fft
import scipy.stats as stats
import scipy as sp
import math as m


# %%
"""parameters for electron momentum simulation"""
m_e = 0.51099895000 * 10**(6) #eV/c2
m_e_err = 0.00000000015 * 10**(6) #eV/c2
detector_width = 6.35 * 10**(-4) #m
detector_length = 1.5 #m
detector_err = 1.1167 * 10**(-4)
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
        self.num_states = fermi_p**2

    def get_p(self, num):
        r, azi, theta = self.gen_pos(num)
        px = r * np.cos(azi) * np.sin(theta)
        py = r * np.sin(azi) * np.sin(theta)
        pz = r * np.cos(theta)

        return np.asarray([px, py, pz])

    def get_pz(self, num):
        r, azi, theta = self.gen_pos(num)
        pz = r * np.cos(theta)
        return pz


    def gen_pos(self, num):
        state = np.random.rand(num)*self.num_states
        r = np.sqrt(state)
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

    def predict_deflection(self, num, a):
        num_rand = np.random.rand(num) * 29
        num_v = (num_rand < a).sum()
        num_c = num_detected - num_v
        pz_c = self.core.get_pz(num_c)
        d_c = self.get_z(pz_c)
        pz_v = self.val.get_pz(num_v)
        d_v = self.get_z(pz_v)
        d = np.concatenate((d_c, d_v))
        return d

# %% Accessory functions
def get_bins(num):
    bins_hist = (np.arange(num) - num // 2) * detector_width
    num = num - 1
    bins_plot = (np.arange(num) - num // 2) * detector_width
    return bins_hist, bins_plot


# %%
core = CoreElectron(scale=core_p_std)
val = ValenceElectron(fermi_p)
i = Interaction(m_e, detector_length)
num = 10**6
pz_val = val.get_pz(num)
pz_core = core.get_pz(num)
d_val = i.get_z(pz_val)
d_core = i.get_z(pz_core)

bins_hist_w, bins_plot_w = get_bins(52)

hist_val = np.histogram(d_val, bins=bins_hist_w)[0]
hist_core = np.histogram(d_core, bins=bins_hist_w)[0]
hist_val = hist_val / np.sum(hist_val)
hist_core = hist_core / np.sum(hist_core)



# %%


fig1, ax1 = plt.subplots()
ax1.bar(bins_plot_w, hist_core, width=detector_width)
ax1.set_xlabel("z [m]")
ax1.set_ylabel("Population Density")
ax1.set_title("Core Electron Photon Deflection")
fig1.tight_layout()
fig1.show()

fig2, ax2 = plt.subplots()
ax2.bar(bins_plot_w, hist_val, width=detector_width)
ax2.set_xlabel("z [m]")
ax2.set_ylabel("Population Density")
ax2.set_title("Valence Electron Photon Deflection")
fig2.tight_layout()
fig2.show()

fig3, ax3 = plt.subplots(2, 1)
ax3[0].bar(bins_plot_w, hist_core, width=detector_width)
ax3[0].set_ylabel("Population Density")
ax3[0].set_title("Core Electron Photon Deflection")
ax3[1].bar(bins_plot_w, hist_val, width=detector_width)
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
d = i.predict_deflection(num_detected, 1.5)

bins_hist, bins_plot = get_bins(52)
hist_d = np.histogram(d, bins=bins_hist)[0]
hist_d = hist_d / np.sum(hist_d)

# %%


fig4, ax4 = plt.subplots()
ax4.bar(bins_plot, hist_d, width=detector_width)
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
total_co_counts = coincident_counts.sum()
total_counts = counts.sum()
mean_counts = np.mean(counts)

adj = (counts - counts[len(counts)//2])/counts[len(counts)//2] + 1
norm_counts = coincident_counts / adj
norm_counts = norm_counts / np.sum(norm_counts)



# %%


fig5, ax5 = plt.subplots()
ax5.bar(bins_plot, coincident_counts, width=detector_width)
ax5.set_xlabel("z [m]")
ax5.set_ylabel("Population Density")
ax5.set_title("Observed deflection of photon with respect to z for Cu")
fig5.tight_layout()
fig5.show()

fig6, ax6 = plt.subplots()
ax6.bar(bins_plot, norm_counts, width=detector_width)
ax6.set_xlabel("z [m]")
ax6.set_ylabel("Population Density")
ax6.set_title("Norm 1: {:.4f}".format(sp.stats.skew(norm_counts)))
fig6.tight_layout()
fig6.show()


# %%

def line(x, a3, a2, a1, a0):
    return a3*x**3 + a2*x**2 + a1*x + a0

a = sp.optimize.curve_fit(line, bins_plot, counts)[0]
a3 = a[0]/a[3]
a2 = a[1]/a[3]
a1 = a[2]/a[3]
a0 = a[3]/a[3]

line_fix = line(bins_plot, a3, a2, a1, a0)

counts_2 = counts / line_fix
co_counts_2 = coincident_counts / line_fix
norm_counts_2 = co_counts_2 / co_counts_2.sum()


fig, ax = plt.subplots()
ax.bar(bins_plot, norm_counts_2, width=detector_width)
ax.set_xlabel("z [m]")
ax.set_ylabel("Population Density")
ax.set_title("Norm 2: {:.4f}".format(sp.stats.skew(norm_counts_2)))
fig.tight_layout()
fig.show()

# %% Correct skew

skew = sp.stats.skew(norm_counts)
shift = 2
shift_arr = np.zeros(2)
norm_counts_3 = norm_counts[:-2*shift]
skew_shift = sp.stats.skew(norm_counts_3)

bins_hist_s, bins_plot_s = get_bins(52 - 2 * shift)
fig, ax = plt.subplots()
ax.bar(bins_plot_s, norm_counts_3, width=detector_width, label="Real Data")
fig.show()

# %%
core = CoreElectron(scale=core_p_std)
val = ValenceElectron(fermi_p)
i = Interaction(m_e, detector_length)
num = 10**6
pz_val = val.get_pz(num)
pz_core = core.get_pz(num)
d_val = i.get_z(pz_val)
d_core = i.get_z(pz_core)

bins_hist_s, bins_plot_s = get_bins(52 - 2 * shift)

hist_val = np.histogram(d_val, bins=bins_hist_s)[0]
hist_core = np.histogram(d_core, bins=bins_hist_s)[0]
hist_val = hist_val / np.sum(hist_val)
hist_core = hist_core / np.sum(hist_core)

fft_val = fft.fft(hist_val)
fft_core = fft.fft(hist_core)



# %%
fft_real = fft.fft(norm_counts_3)
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
fft_real = fft.fft(norm_counts_3)
b = np.mean(fft_real - fft_core) / np.mean(fft_val - fft_core)
a = 1-b
print("a guess: {:.2f}".format(a))

z_fft = a.real * hist_core + b.real * hist_val

fig, ax = plt.subplots()
ax.bar(bins_plot_s, norm_counts_3, width=detector_width, label='Real Data')
ax.bar(bins_plot_s, z_fft, width=detector_width, label='FFT Prediction', fill=False)
ax.set_title("Valence ratio: {:.3f}".format(a.real))
ax.legend()
fig.show()

# %%
b = np.mean(norm_counts_3 - hist_core) / np.mean(hist_val - hist_core)
a = 1-b
z = b * hist_core + a * hist_val
print("a guess: {:.2f}".format(a))

fig, ax = plt.subplots()
ax.bar(bins_plot_s, norm_counts_3, width=detector_width, label='Real Data')
ax.bar(bins_plot_s, z, width=detector_width, label='Linear Prediction', fill=False)
ax.set_title("Valence ratio: {:.3f}".format(a.real))
ax.legend()
fig.show()

# %%
a = np.linspace(9, 10, num=100)
num_detected = int(total_counts)

core = CoreElectron(scale=core_p_std)
val = ValenceElectron(fermi_p)
i = Interaction(core, val)
pred = np.zeros((a.shape[0], norm_counts_3.shape[0]))
diff = np.zeros(a.shape)

for idx in range(len(a)):
    d = i.predict_deflection(num_detected, a[idx])
    hist_d = np.histogram(d, bins=bins_hist_s)[0]
    pred[idx] = hist_d / np.sum(hist_d)
    temp_diff = pred[idx] - norm_counts_3
    diff[idx] = np.sqrt(temp_diff.dot(temp_diff))

min = np.argsort(diff, axis=0)


# %%

num = pred.shape[0]

dim = int(m.ceil(np.sqrt(pred.shape[0])))


fig, ax = plt.subplots()
ax.plot(a, diff)
fig.show()


fig, ax = plt.subplots(m.ceil(dim), m.ceil(dim))
fig.set_size_inches(50,50)
for i in range(num):

    ax[i // dim][i % dim].bar(bins_plot_s, norm_counts_3, width=detector_width, label="Real Data")
    ax[i // dim][i % dim].bar(bins_plot_s, pred[i], width=detector_width, label="Prediction {:.2f}".format(a[i]), fill=False)
    ax[i // dim][i % dim].set_title("Prediction {:.2f}".format(a[i]))


fig.show()


# %%

fig9, ax9 = plt.subplots()
ax9.bar(bins_plot_s, norm_counts_3, width=detector_width, label="Real Data")
ax9.bar(bins_plot_s, pred[min[0]], width=detector_width, label="Best Prediction", fill=False, edgecolor='red')
ax9.bar(bins_plot_s, z_fft, width=detector_width, label="Best FFT Prediction", fill=False, edgecolor='green')
ax9.legend()
ax9.set_xlabel("z [m]")
ax9.set_ylabel("Population Density")
ax9.set_title("Observed deflection of photon with respect to z for Cu")
fig9.tight_layout()
fig9.show()

# %%

core = CoreElectron(scale=core_p_std)
val = ValenceElectron(fermi_p)
i = Interaction(core, val)
pred2 = i.predict_deflection(num_detected, a=a[min[0]])
hist_pred = np.histogram(pred2, bins=bins_hist)[0]
hist_pred = hist_pred / np.sum(hist_pred)


fig, ax = plt.subplots()
ax.bar(bins_plot, norm_counts, width=detector_width, label="Real Data")
ax.bar(bins_plot, hist_pred, width=detector_width, label="Best Prediction", fill=False)
ax.legend()


ax.set_xlabel("z [m]")
ax.set_ylabel("Population Density")
ax.set_title("Observed deflection of photon with respect to z for Cu")
fig.tight_layout()
fig.show()


# %%





















