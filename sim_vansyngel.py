import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.cm as cm

nside = 256
l0 = 70.
b0 = 24.
fM0 = 0.9
p0 = 0.26
Nlayer = 1
alphaM0 = -2.5
sigma_beta0 = 0.15
gamma0 = -3.
beta0 = 1.6
temp0 = 19.6
nu0_dust = 353.




def fcmb(nu):
    """ CMB SED (in antenna temperature units).
    """
    x=0.017608676067552197*nu
    ex=np.exp(x)
    return ex*(x/(ex-1))**2


def comp_sed(nu, nu0, beta, temp, typ):
    """ Component SEDs (in antenna temperature units).
    """
    if typ=='cmb':
        return fcmb(nu)
    elif typ=='dust':
        x_to=0.04799244662211351*nu/temp
        x_from=0.04799244662211351*nu0/temp
        return (nu/nu0)**(1+beta)*(np.exp(x_from)-1)/(np.exp(x_to)-1)
    elif typ=='sync':
        return (nu/nu0)**beta
    return None


def get_beta(ns=nside, beta=beta0, sigma=sigma_beta0, gamma=gamma0):
    npix = hp.nside2npix(ns)
    beta_map = np.ones(npix) * beta
    if sigma > 0:
        ls = np.arange(3*ns)
        cls = np.zeros(3*ns)
        cls[2:] = ls[2:]**gamma0
        dbeta = hp.synfast(cls, ns, new=True, verbose=False)
        norm = sigma / np.std(dbeta)
        beta_map += norm * dbeta
    return beta_map
    

def get_B0(b=b0, l=l0):
    theta = np.radians(90-b)
    phi = np.radians(l)
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    return np.array([st*cp, st*sp, ct])


def get_Bt(ns=nside, alphaM=alphaM0):
    ls = np.arange(3*ns)
    cls = np.zeros(3*ns)
    cls[2:] = ls[2:]**alphaM
    bt = np.array([hp.synfast(cls, ns, new=True, verbose=False)
                   for i in range(3)])
    btmod = np.sqrt(np.sum(bt**2, axis=0))
    bt = bt[:, :]/btmod[None, :]
    return bt


def get_B(ns=nside, alphaM=alphaM0, b=b0, l=l0, fM=fM0):
    bt = get_Bt(ns=ns, alphaM=alphaM)
    b0 = get_B0(b=b, l=l)
    return b0[:, None] + fM*bt


def get_gamma_phi(ns=nside, alphaM=alphaM0, b=b0, l=l0, fM=fM0):
    npix = hp.nside2npix(ns)

    # Get unit magnetic field direction
    bM = get_B(ns=ns, alphaM=alphaM, b=b, l=l, fM=fM)
    eb = bM / np.sqrt(np.sum(bM**2, axis=0))

    # Get local tetrad
    th, ph = hp.pix2ang(nside, np.arange(npix))
    ct = np.cos(th)
    st = np.sin(th)
    cp = np.cos(ph)
    sp = np.sin(ph)
    er = np.array([st*cp, st*sp, ct])
    et = np.array([ct*cp, ct*sp, -st])
    ep = np.array([-sp, cp, np.zeros(npix)])

    # LOS angle
    sin_gamma = np.sum(eb * er, axis=0)
    cos2_gamma = 1 - sin_gamma**2

    # Polarization angle
    # Fig. 14 of https://arxiv.org/pdf/1405.0872.pdf
    # First, compute transverse magnetic field
    b_trans = eb - er * sin_gamma[None, :]
    b_trans_mod = np.sqrt(np.sum(b_trans**2, axis=0))
    bmsk0 = b_trans_mod > 0
    b_trans[:, bmsk0] = b_trans[:, bmsk0] / b_trans_mod[None, bmsk0]
    # Dot with e_phi and e_theta to get chi
    cos_chi = -np.sum(b_trans * ep, axis=0)
    sin_chi = np.sum(b_trans * et, axis=0)
    chi = np.arctan2(sin_chi, cos_chi)
    phi = chi - np.pi/2
    return cos2_gamma, phi


def get_qu_maps(nu, ns=nside, alphaM=alphaM0, b=b0, l=l0, fM=fM0, p=p0,
                sigma=sigma_beta0, gamma=gamma0, beta=beta0, nu0=nu0_dust,
                temp=temp0):
    npix = hp.nside2npix(nside)
    q = np.zeros(npix)
    u = np.zeros(npix)
    for i in range(Nlayer):
        c2g, phi = get_gamma_phi(ns=ns, alphaM=alphaM,
                                 b=b, l=l, fM=fM)
        beta = get_beta(ns=ns, beta=beta, sigma=sigma, gamma=gamma)
        sed = comp_sed(nu, nu0=nu0, beta=beta,
                       temp=temp, typ='dust')
        q += sed * c2g * np.cos(2*phi)
        u += sed * c2g * np.sin(2*phi)
    return p*q, p*u


c2g, p = get_gamma_phi()
q, u = get_qu_maps(280.)

hp.mollview(c2g)
hp.mollview(p/(2*np.pi), cmap=cm.twilight)
hp.mollview(q)
hp.mollview(u)
plt.figure()
plt.hist(q, bins=50, histtype='step', color='r')
plt.hist(u, bins=50, histtype='step', color='b')
plt.show()
