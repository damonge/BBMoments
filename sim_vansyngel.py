import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import utils_vansyngel as uv
import utils as ut
import pymaster as nmt
import os

nu0_dust = 353.
beta0 = 1.6
temp0 = 19.6
nu_use = 39.

nside = 256
np.random.seed(1234)

msk = hp.ud_grade(hp.read_map("./data/norm_nHits_SA_35FOV.fits", verbose=False),
                  nside_out=nside)
b = nmt.NmtBin(nside, nlb=10, is_Dell=True)
w = nmt.NmtWorkspace()
if os.path.isfile("wsp_ns%d.fits" % nside):
    w.read_from("wsp_ns%d.fits" % nside)
else:
    f = nmt.NmtField(msk, [msk, msk], n_iter=0)
    w.compute_coupling_matrix(f, f, b)
    w.write_to("wsp_ns%d.fits" % nside)

qb, ub = hp.read_map("./data/dust_QU_equatorial.fits", verbose=False, field=[0, 1])
qb = hp.ud_grade(qb, nside_out=nside)
ub = hp.ud_grade(ub, nside_out=nside)
beta_map = uv.get_beta(nside, beta0, 0.2, -3.)
sed = ut.comp_sed(nu_use, nu0_dust, beta_map, temp0, 'dust')
qb *= sed
ub *= sed

nus = np.array([29., 39., 95., 145., 220., 280.])
q, u = uv.get_dust_sim(nus, nside)
q = q[1]
u = u[1]

f = nmt.NmtField(msk, [q, u], n_iter=0)
fb = nmt.NmtField(msk, [qb, ub], n_iter=0)
cl = w.decouple_cell(nmt.compute_coupled_cell(f, f))
clb = w.decouple_cell(nmt.compute_coupled_cell(fb, fb))

plt.figure()
leff = b.get_effective_ells()
plt.plot(leff, cl[0], 'k-')
plt.plot(leff, clb[0], 'r-')
plt.plot(leff, cl[3], 'k--')
plt.plot(leff, clb[3], 'r--')
plt.loglog()


hp.mollview(q*msk)
hp.mollview(u*msk)
hp.mollview(qb*msk)
hp.mollview(ub*msk)
plt.figure()
plt.hist(q*msk, bins=50, range=[-10, 10], histtype='step', color='k')
plt.hist(qb*msk, bins=50, range=[-10, 10], histtype='step', color='r')
plt.figure()
plt.hist(u*msk, bins=50, range=[-10, 10], histtype='step', color='k')
plt.hist(ub*msk, bins=50, range=[-10, 10], histtype='step', color='r')
plt.show()
