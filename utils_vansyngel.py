import numpy as np
import healpy as hp
import utils as ut


def get_beta(ns, beta, sigma, gamma):
    npix = hp.nside2npix(ns)
    beta_map = np.ones(npix) * beta
    if sigma > 0:
        ls = np.arange(3*ns)
        cls = np.zeros(3*ns)
        cls[2:] = ls[2:]**gamma
        dbeta = hp.synfast(cls, ns, new=True, verbose=False)
        norm = sigma / np.std(dbeta)
        beta_map += norm * dbeta
    return beta_map
    

def get_B0(b, l):
    rot = hp.Rotator(coord=['G', 'C'])
    theta = np.radians(90-b)
    phi = np.radians(l)
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    return rot(np.array([st*cp, st*sp, ct]))


def get_Bt(ns, alphaM):
    ls = np.arange(3*ns)
    cls = np.zeros(3*ns)
    cls[2:] = ls[2:]**alphaM
    bt = np.array([hp.synfast(cls, ns, new=True, verbose=False)
                   for i in range(3)])
    btmod = np.sqrt(np.sum(bt**2, axis=0))
    bt = bt[:, :]/btmod[None, :]
    return bt


def get_B(ns, alphaM, b, l, fM):
    bt = get_Bt(ns=ns, alphaM=alphaM)
    b0 = get_B0(b=b, l=l)
    return b0[:, None] + fM*bt


def get_gamma_phi(ns, alphaM, b, l, fM):
    npix = hp.nside2npix(ns)

    # Get unit magnetic field direction
    bM = get_B(ns=ns, alphaM=alphaM, b=b, l=l, fM=fM)
    eb = bM / np.sqrt(np.sum(bM**2, axis=0))

    # Get local tetrad
    th, ph = hp.pix2ang(ns, np.arange(npix))
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


def get_qu_maps(nu, ns, alphaM, b, l, fM, p, sigma, gamma, beta, nu0, temp, Nly):
    npix = hp.nside2npix(ns)
    if np.ndim(nu) == 0:
        nu_use = np.atleast_1d(nu)
    else:
        nu_use = nu

    n_nu = len(nu_use)
    q = np.zeros([n_nu, npix])
    u = np.zeros([n_nu, npix])
    for i in range(Nly):
        print(i)
        c2g, phi = get_gamma_phi(ns=ns, alphaM=alphaM,
                                 b=b, l=l, fM=fM)
        beta = get_beta(ns=ns, beta=beta, sigma=sigma, gamma=gamma)
        for i_f, f in enumerate(nu_use):
            sed = ut.comp_sed(f, nu0, beta,
                              temp, 'dust')
            q[i_f, :] += sed * c2g * np.cos(2*phi)
            u[i_f, :] += sed * c2g * np.sin(2*phi)

    if np.ndim(nu) == 0:
        q = np.squeeze(q)
        u = np.squeeze(u)
    return p*q, p*u


def get_dust_sim(nu, ns, alphaM=-2.5, b=24., l=70., fM=0.9, p=0.037,
                 sigma=0.13, gamma=-3., beta=1.6, nu0=353., temp=19.6,
                 Nly=7): 
    tb = hp.ud_grade(hp.read_map("./data/dust_T_equatorial.fits", verbose=False),
                     nside_out=ns)
    ls = np.arange(3*ns)
    tb = hp.alm2map(hp.almxfl(hp.map2alm(tb),
                              1./(1.+(ls/500)**2)),
                    nside=ns, verbose=False)
    tb *= ut.comp_sed(nu0, 545., beta, temp, 'dust')

    if np.ndim(nu) == 0:
        nu_use = np.atleast_1d(nu)
    else:
        nu_use = nu
    q, u = get_qu_maps(nu_use, ns, alphaM, b, l, fM, p,
                       sigma, gamma, beta, nu0, temp, Nly)
    q = q[:, :] * tb[None, :]
    u = u[:, :] * tb[None, :]

    if np.ndim(nu) == 0:
        q = np.squeeze(q)
        u = np.squeeze(u)
    return q, u
