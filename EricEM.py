#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
import numpy as np
import matplotlib.pyplot as plt


# %%
m_e = 0.51099895000
m_e_err = 0.00000000015
fermi_p = 2.675
core_mean = 4
c = 1

# %%
class CoreElectron():
    def __init__(self, loc=0.0, scale=1.0):
        self.loc = loc
        self.scale = scale

    def get_pz(self, num):
        pz = np.random.normal(loc=self.loc, scale=self.scale, size=num)
        return pz

# %% Valence Electrons
class ValenceElectron():
    def __init__(self, fermi_p):
        self.fermi_p = fermi_p

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
        r = np.random.rand(num)*self.fermi_p
        theta = np.random.rand(num) * np.pi
        azi = 2 * np.random.rand(num) * np.pi
        return r, azi, theta

# %%
class Interaction():
    def __init__(self, m_e, l=1.50):
        self.m_e = m_e
        self.l = l
        self.c = 1

    def get_E(self, pz):
        E2 = pz**2 + 2 * self.m_e**2
        E = E2**0.5
        return E

    def get_deflection(self, pz):
        E = self.get_E(pz)
        frac = self.c*pz/E
        defl = np.arctan(frac)*2
        return defl

    def get_z(self, pz):
        defl = self.get_deflection(pz)
        d = self.l * defl
        return d





# %%
c = CoreElectron(scale=core_mean)
v = ValenceElectron(fermi_p)
i = Interaction(m_e)
num = 100000
pz_val = v.get_pz(num)
pz_core = c.get_pz(num)
d_val = i.get_z(pz_val)
d_core = i.get_z(pz_core)
print(pz_core)
fig, ax = plt.subplots(2, 1)
ax[0].hist(d_core, bins=50, density=True)
ax[1].hist(d_val, bins=50, density=True)
fig.show()
