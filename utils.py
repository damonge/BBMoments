import healpy as hp
import numpy as np
from scipy.special import zeta
import pysm
from pysm.nominal import models
import os
from pyshtools.utils import Wigner3j
from noise_calc import Simons_Observatory_V3_SA_noise,Simons_Observatory_V3_SA_beams

def get_w3j(lmax=383):
    """ Calculates Wigner 3J symbols (or reads them from file if they already exist).
    """
    if os.path.isfile('data/w3j_lmax%d.npz' % lmax):
        big_w3j = np.load('data/w3j_lmax%d.npz' % lmax)['w3j']
    else:
        ells_w3j = np.arange(lmax+1)
        w3j = np.zeros_like(ells_w3j, dtype=float)
        big_w3j = np.zeros((lmax+1, lmax+1, lmax+1))
        for ell1 in ells_w3j[1:]:
            for ell2 in ells_w3j[1:]:
                w3j_array, ellmin, ellmax = Wigner3j(ell1, ell2, 0, 0, 0 )
                w3j_array = w3j_array[:ellmax - ellmin + 1]
                #make the w3j_array the same shape as the w3j
                if len(w3j_array) < len(ells_w3j):
                    reference = np.zeros(len(w3j))
                    reference[:w3j_array.shape[0]] = w3j_array
                    w3j_array = reference
                
                w3j_array = np.concatenate([w3j_array[-ellmin:],w3j_array[:-ellmin]])
                w3j_array = w3j_array[:len(ells_w3j)]
                w3j_array[:ellmin] = 0
            
                big_w3j[:,ell1,ell2] = w3j_array
            
        big_w3j = big_w3j**2
        np.savez("data/w3j_lmax%d" % lmax, w3j = big_w3j)
    return big_w3j

def get_vector_and_covar(ls, cls, fsky=1.):
    """ Vectorizes an array of C_ells and computes their
    associated covariance matrix.

    Args:
        ls: array of multipole values.
        cls: array of power spectra with shape [nfreq, npol, nfreq, npol, nell]

    Returns:
        translator: an array of shape [nfreq*npol, nfreq*npol] that contains
            the vectorized indices for a given pair of map indices.
        cl_vec: vectorized power spectra. Shape [n_pairs, nell]
        cov: vectorized covariance. Shape [n_pairs, n_ell, n_pair, n_ell]
    """
    nfreq, npol, _, _, nls = cls.shape
    nmaps = nfreq*npol
    ind = np.triu_indices(nmaps)
    nx = len(ind[0])
    # 2D to 1D translator
    translator = np.zeros([nmaps, nmaps], dtype=int)
    ix = 0
    for i1 in range(nmaps):
        for i2 in range(i1, nmaps):
            translator[i1, i2] = ix
            if i1 != i2:
                translator[i2, i1] = ix
            ix += 1

    delta_ell = np.mean(np.diff(ls))
    fl = 1./((2*ls+1)*delta_ell*fsky)
    cov = np.zeros([nx, nls, nx, nls])
    cl_vec = np.zeros([nx, nls])
    cl_maps = cls.reshape([nmaps, nmaps, nls])
    for ii, (i1, i2) in enumerate(zip(ind[0], ind[1])):
        cl_vec[ii, :] = cl_maps[i1, i2, :]
        for jj, (j1, j2) in enumerate(zip(ind[0], ind[1])):
            covar = (cl_maps[i1, j1, :] * cl_maps[i2, j2, :] +
                     cl_maps[i1, j2, :] * cl_maps[i2, j1, :]) * fl
            cov[ii, :, jj, :] = np.diag(covar)
    return translator, cl_vec, cov


def bin_cls(cls, delta_ell=10):
    """ Returns a binned-version of the power spectra.
    """
    nls = cls.shape[-1]
    ells = np.arange(nls)
    delta_ell = 10
    N_bins = (nls-2)//delta_ell
    w = 1./delta_ell
    W = np.zeros([N_bins, nls])
    for i in range(N_bins):
        W[i, 2+i*delta_ell:2+(i+1)*delta_ell] = w
    l_eff = np.dot(ells, W.T)
    cl_binned = np.dot(cls, W.T)
    return l_eff, W, cl_binned


def map2cl(maps, maps2=None, iter=0):
    """ Returns an array with all auto- and cross-correlations
    for a given set of Q/U frequency maps.

    Args:
        maps: set of frequency maps with shape [nfreq, 2, npix].
        maps2: set of frequency maps with shape [nfreq, 2, npix] to cross-correlate with.
        iter: iter parameter for anafast (default 0).

    Returns:
        Set of power spectra with shape [nfreq, 2, nfreq, 2, n_ell].
    """
    nfreq, npol, npix = maps.shape
    nside = hp.npix2nside(npix)
    nls = 3*nside
    ells = np.arange(nls)
    cl2dl = ells*(ells+1)/(2*np.pi)
    if maps2 is None:
        maps2 = maps

    cl_out = np.zeros([nfreq, npol, nfreq, npol, nls])
    for i in range(nfreq):
        m1 = np.zeros([3, npix])
        m1[1:,:]=maps[i, :, :]
        for j in range(i,nfreq):
            m2 = np.zeros([3, npix])
            m2[1:,:]=maps2[j, :, :]

            cl = hp.anafast(m1, m2, iter=0)
            cl_out[i, 0, j, 0] = cl[1] * cl2dl
            cl_out[i, 1, j, 1] = cl[2] * cl2dl
            if j!=i:
                cl_out[j, 0, i, 0] = cl[1] * cl2dl
                cl_out[j, 1, i, 1] = cl[2] * cl2dl
    return cl_out


def get_freqs():
    """ Return 6 SO frequencies.
    """
    return np.array([27., 39., 93., 145., 225., 280.]) 


def fcmb(nu):
    """ CMB SED (in antenna temperature units).
    """
    x=0.017608676067552197*nu
    ex=np.exp(x)
    return ex*(x/(ex-1))**2


def comp_sed(nu,nu0,beta,temp,typ):
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

def w3j_sandwich(ls, cl1, cl2, w3j):
    """ Computes:
    sum_{l1, l2} ((2l1+1) * (2l2+1) * cl1_l1 * cl2_l2 * w3j[l, l1, l2]) / 4pi
    """
    nl = min(len(w3j), len(cl1))
    nl_out = len(cl1)
    v_left = ((2*ls+1)*cl1)[:nl]
    v_right = ((2*ls+1)*cl2)[:nl]
    mat = w3j[:nl][:,:nl][:,:,:nl]
    sand = np.dot(np.dot(mat, v_right), v_left) / (4 * np.pi)
    v_out = np.zeros(nl_out)
    v_out[:nl] = sand
    return v_out

def get_delta_beta_cl(amp, gamma, ls, l0=80., l_cutoff=2):
    """
    Returns power spectrum for spectral index fluctuations.

    Args:
        amp: amplitude
        gamma: tilt
        ls: array of ells
        l0: pivot scale (default: 80)
        l_cutoff: ell below which the power spectrum will be zero.
            (default: 2).

    Returns:
        Array of Cls
    """
    ind_above = np.where(ls > l_cutoff)[0]
    cls = np.zeros(len(ls))
    cls[ind_above] = amp * (ls[ind_above] / l0)**gamma
    return cls


def get_delta_beta_amp(sigma, gamma):
    """
    Returns power spectrum amplitude for beta fluctuations that
    should achieve a given standard deviation. Assumes l_cutoff=2.

    Args:
        sigma: requested standard deviation.
        gamma: tilt

    Returns:
        Spectral index power spectrum amplitude.
    """
    return 4*np.pi*sigma**2*80**gamma/(-3+2*zeta(-1-gamma)+zeta(-gamma))


def get_beta_map(nside, beta0, amp, gamma, l0=80, l_cutoff=2, seed=None):
    """
    Returns realization of the spectral index map.

    Args:
        nside: HEALPix resolution parameter.
        beta0: mean spectral index.
        amp: amplitude
        gamma: tilt
        l0: pivot scale (default: 80)
        l_cutoff: ell below which the power spectrum will be zero.
            (default: 2).
        seed: seed (if None, a random seed will be used).

    Returns:
        Spectral index map
    """
    if seed is not None:
        np.random.seed(seed)
    ls = np.arange(3*nside)
    cls = get_delta_beta_cl(amp, gamma, ls, l0, l_cutoff)
    mp = hp.synfast(cls, nside, verbose=False)
    mp += beta0
    return mp


def get_default_params():
    """ Returns default set of parameters describing a
    given sky realization. The parameters are distributed
    into 2 dictionaries, corresponding to "mean"
    (i.e. non-moment-related) parameters and "moment" parameters.

    The mean parameters are:
      - 'A_dust_BB': B-mode dust power spectrum amplitude at
          the frequency `nu0_dust_def'.
      - 'EB_dust': EE to BB ratio.
      - 'alpha_dust_EE': tilt in D_l^EE for dust
      - 'alpha_dust_BB': tilt in D_l^BB for dust
      - 'nu0_dust_def': frequency at which 'A_dust_BB' is defined.
      - 'nu0_dust': frequency at which amplitude maps should be
          generated. At this frequency spectral index variations
          are irrelevant.
      - 'beta_dust': mean dust spectral index.
      - A copy of all the above for synchrotron (called `sync`
          instead of `dust`).
      - 'temp_dust': dust temperature.
      - 'include_XXX': whether to include component XXX, where
          XXX is CMB, sync or dust.
      - 'include_Y': whether to include Y polarization, where 
          Y is E or B. 

    The moment parameters are:
      - 'amp_beta_dust': delta_beta power spectrum amplitude
          for dust.
      - 'gamma_beta_dust': delta_beta power spectrum tilt.
          for dust.
      - 'l0_beta_dust': pivot scale for delta_beta (80).
      - 'l_cutoff_beta_dust': minimum ell for which the delta
          beta power spectrum is non-zero (2).
      - A copy of the above for synchrotron.
    """
    mean_pars = {'A_dust_BB': 5,
                 'EB_dust': 2.,
                 'alpha_dust_EE': -0.42,
                 'alpha_dust_BB': -0.42,
                 'nu0_dust': 353.,
                 'nu0_dust_def': 353.,
                 'beta_dust': 1.6,
                 'temp_dust': 19.6,
                 'A_sync_BB': 2,
                 'EB_sync': 2.,
                 'alpha_sync_EE': -0.6,
                 'alpha_sync_BB': -0.6,
                 'nu0_sync': 23.,
                 'nu0_sync_def': 23.,
                 'beta_sync': -3.,
                 'A_lens' : 1,
                 'include_CMB': True,
                 'include_dust': True,
                 'include_sync': True,
                 'include_E': True,
                 'include_B': True,
    }
    moment_pars = {'amp_beta_sync': 0.,
                   'gamma_beta_sync': -3.,
                   'l0_beta_sync': 80.,
                   'l_cutoff_beta_sync': 2,
                   'amp_beta_dust': 0.,
                   'gamma_beta_dust': -3.,
                   'l0_beta_dust': 80.,
                   'l_cutoff_beta_dust': 2}
    return mean_pars, moment_pars


def get_mean_spectra(lmax, mean_pars):
    """ Computes amplitude power spectra for all components
    """
    ells = np.arange(lmax+1)
    dl2cl = np.ones(len(ells))
    dl2cl[1:] = 2*np.pi/(ells[1:]*(ells[1:]+1.))
    cl2dl = (ells*(ells+1.))/(2*np.pi)

    # Translate amplitudes to reference frequencies
    A_dust_BB = mean_pars['A_dust_BB'] * (comp_sed(mean_pars['nu0_dust'],
                                                   mean_pars['nu0_dust_def'],
                                                   mean_pars['beta_dust'],
                                                   mean_pars['temp_dust'],
                                                   'dust'))**2
    A_sync_BB = mean_pars['A_sync_BB'] * (comp_sed(mean_pars['nu0_sync'],
                                                   mean_pars['nu0_sync_def'],
                                                   mean_pars['beta_sync'],
                                                   None, 'sync'))**2
    # Dust amplitudes
    A_dust_BB = A_dust_BB * fcmb(mean_pars['nu0_dust'])**2
    dl_dust_bb = A_dust_BB * ((ells+1E-5) / 80.)**mean_pars['alpha_dust_BB']
    dl_dust_ee = mean_pars['EB_dust'] * A_dust_BB * \
                 ((ells+1E-5) / 80.)**mean_pars['alpha_dust_EE']
    cl_dust_bb = dl_dust_bb * dl2cl
    cl_dust_ee = dl_dust_ee * dl2cl
    if not mean_pars['include_E']:
        cl_dust_ee *= 0 
    if not mean_pars['include_B']:
        cl_dust_bb *= 0
    if not mean_pars['include_dust']:
        cl_dust_bb *= 0
        cl_dust_ee *= 0

    # Sync amplitudes
    A_sync_BB = A_sync_BB * fcmb(mean_pars['nu0_sync'])**2
    dl_sync_bb = A_sync_BB * ((ells+1E-5) / 80.)**mean_pars['alpha_sync_BB']
    dl_sync_ee = mean_pars['EB_sync'] * A_sync_BB * \
                 ((ells+1E-5) / 80.)**mean_pars['alpha_sync_EE']
    cl_sync_bb = dl_sync_bb * dl2cl
    cl_sync_ee = dl_sync_ee * dl2cl
    if not mean_pars['include_E']:
        cl_sync_ee *= 0 
    if not mean_pars['include_B']:
        cl_sync_bb *= 0
    if not mean_pars['include_sync']:
        cl_sync_bb *= 0
        cl_sync_ee *= 0

    # CMB amplitude
    l,dtt,dee,dbb,dte=np.loadtxt("data/camb_lens_nobb.dat",unpack=True)
    l = l.astype(int)
    msk = l <= lmax
    l = l[msk]
    dltt=np.zeros(len(ells)); dltt[l]=dtt[msk]
    dlee=np.zeros(len(ells)); dlee[l]=dee[msk]
    dlbb=np.zeros(len(ells)); dlbb[l]=dbb[msk]
    dlte=np.zeros(len(ells)); dlte[l]=dte[msk]  
    cl_cmb_bb=dlbb * dl2cl
    cl_cmb_ee=dlee * dl2cl
    if not mean_pars['include_E']:
        cl_cmb_ee *= 0 
    if not mean_pars['include_B']:
        cl_cmb_bb *= 0
    if not mean_pars['include_CMB']:
        cl_cmb_bb *= 0
        cl_cmb_ee *= 0

    return (ells, dl2cl, cl2dl,
            cl_dust_bb, cl_dust_ee,
            cl_sync_bb, cl_sync_ee,
            cl_cmb_bb, cl_cmb_ee)


def get_theory_sacc(nside, mean_pars=None, moment_pars=None, delta_ell=10, add_11=False, add_02=False):
    """ Generate a SACC object containing a set of theory power spectra.

        nside: HEALPix resolution parameter.
        seed: seed to be used (if `None`, then a random seed will
            be used).
        mean_pars: mean parameters (see `get_default_params`).
            If `None`, then a default set will be used.
        moment_pars: mean parameters (see `get_default_params`).
            If `None`, then a default set will be used.
        delta_ell: bandpower size to use (default 10).

    Returns:
        A dictionary containing power spectrum information.
    """
    import sacc
    nus = get_freqs()
    nfreq = len(nus)
    th = get_theory_spectra(nside, mean_pars=mean_pars, moment_pars=moment_pars,
                            delta_ell=delta_ell, add_11=add_11, add_02=add_02)
    l_eff = th['ls_binned']

    # Binning
    typ, ell, t1, q1, t2, q2 = [], [], [], [], [], []
    for b1 in range(nfreq):
        for b2 in range(b1, nfreq):
            if (b1==b2):
                types = ['EE', 'EB', 'BB']
            else:
                types = ['EE', 'EB', 'BE', 'BB']
            for ty in types:
                for il, l in enumerate(l_eff):
                    ell.append(l)
                    typ.append(ty)
                    t1.append(b1)
                    t2.append(b2)
                    q1.append('C')
                    q2.append('C')
    bnn = sacc.Binning(typ,ell,t1,q1,t2,q1)

    # Tracers
    trs = []
    for inu, nu in enumerate(nus):
        fs = np.array([nu-1., nu, nu+1])
        bs = np.array([0., 1., 0.])
        T=sacc.Tracer('band%d'%(inu+1), 'CMBP',
                      fs, bs, exp_sample='SO_SAT')
        T.addColumns({'dnu':np.ones(3)})
        trs.append(T)

    # SACC
    clv = th['cls_binned'].flatten()
    sacc_mean = sacc.MeanVec(clv)
    sacc_prec = sacc.Precision(matrix=th['cov_binned'].reshape([len(clv),len(clv)]),
                               is_covariance=True,mode="dense")
    s = sacc.SACC(trs, bnn, mean=sacc_mean, precision=sacc_prec)
    return s


def get_theory_spectra(nside, mean_pars=None, moment_pars=None, delta_ell=10, add_11=False, add_02=False):
    """ Generate a set of theory power spectra for set of input sky parameters.

    Args:
        nside: HEALPix resolution parameter.
        seed: seed to be used (if `None`, then a random seed will
            be used).
        mean_pars: mean parameters (see `get_default_params`).
            If `None`, then a default set will be used.
        moment_pars: mean parameters (see `get_default_params`).
            If `None`, then a default set will be used.
        delta_ell: bandpower size to use (default 10).

    Returns:
        A dictionary containing power spectrum information.
    """
    nu = get_freqs()
    npix = hp.nside2npix(nside)
    if mean_pars is None:
        mean_pars, _ = get_default_params()
    if moment_pars is None:
        _, moment_pars = get_default_params()
    lmax = 3*nside-1

    # Power spectra
    ells, dl2cl, cl2dl, cl_dust_bb, cl_dust_ee, cl_sync_bb, cl_sync_ee, cl_cmb_bb, cl_cmb_ee = get_mean_spectra(lmax, mean_pars)

    # Frequency spectra
    f_dust = comp_sed(nu, mean_pars['nu0_dust'], mean_pars['beta_dust'],
                      mean_pars['temp_dust'], 'dust')
    f_sync = comp_sed(nu, mean_pars['nu0_sync'], mean_pars['beta_sync'],
                      None, 'sync')
    f_cmb = fcmb(nu)
    f_dust /= f_cmb
    f_sync /= f_cmb
    f_cmb /= f_cmb

    # Background spectra
    C_ells_sky = np.zeros([6, 2, 6, 2, len(ells)])
    C_ells_sky[:, 0, :, 0, :] = (cl_cmb_ee[None, None, :] * np.outer(f_cmb, f_cmb)[:, :, None] +
                                 cl_dust_ee[None, None, :] * np.outer(f_dust, f_dust)[:, :, None] +
                                 cl_sync_ee[None, None, :] * np.outer(f_sync, f_sync)[:, :, None])
    C_ells_sky[:, 1, :, 1, :] = (cl_cmb_bb[None, None, :] * np.outer(f_cmb, f_cmb)[:, :, None] +
                                 cl_dust_bb[None, None, :] * np.outer(f_dust, f_dust)[:, :, None] +
                                 cl_sync_bb[None, None, :] * np.outer(f_sync, f_sync)[:, :, None])

    # Add moments
    w3j = get_w3j()
    cl_beta_dust = get_delta_beta_cl(moment_pars['amp_beta_dust'],
                                     moment_pars['gamma_beta_dust'],
                                     ells, moment_pars['l0_beta_dust'],
                                     moment_pars['l_cutoff_beta_dust'])
    cl_beta_sync = get_delta_beta_cl(moment_pars['amp_beta_sync'],
                                     moment_pars['gamma_beta_sync'],
                                     ells, moment_pars['l0_beta_sync'],
                                     moment_pars['l_cutoff_beta_sync'])
    cl_dust_1x1_bb = w3j_sandwich(ells, cl_dust_bb, cl_beta_dust, w3j)
    cl_dust_1x1_ee = w3j_sandwich(ells, cl_dust_ee, cl_beta_dust, w3j)
    cl_sync_1x1_bb = w3j_sandwich(ells, cl_sync_bb, cl_beta_sync, w3j)
    cl_sync_1x1_ee = w3j_sandwich(ells, cl_sync_ee, cl_beta_sync, w3j)
    sig2_beta_dust = np.sum((2*ells+1)*cl_beta_dust)/(4*np.pi)
    sig2_beta_sync = np.sum((2*ells+1)*cl_beta_sync)/(4*np.pi)
    cl_dust_0x2_bb = sig2_beta_dust * cl_dust_bb
    cl_dust_0x2_ee = sig2_beta_dust * cl_dust_ee
    cl_sync_0x2_bb = sig2_beta_sync * cl_sync_bb
    cl_sync_0x2_ee = sig2_beta_sync * cl_sync_ee
    # Derivatives of spectrum
    x_dust = np.log(nu / mean_pars['nu0_dust'])
    f_dust_1 = x_dust * f_dust
    f_dust_2 = x_dust**2 * f_dust
    x_sync = np.log(nu / mean_pars['nu0_sync'])
    f_sync_1 = x_sync * f_sync
    f_sync_2 = x_sync**2 * f_sync

    if add_11:
        # 1x1
        f_dust_1x1 = np.outer(f_dust_1, f_dust_1)[:, :, None]
        f_sync_1x1 = np.outer(f_sync_1, f_sync_1)[:, :, None]
        C_ells_sky[:, 0, :, 0, :] += (cl_dust_1x1_ee[None, None, :] * f_dust_1x1 +
                                      cl_sync_1x1_ee[None, None, :] * f_sync_1x1)
        C_ells_sky[:, 1, :, 1, :] += (cl_dust_1x1_bb[None, None, :] * f_dust_1x1 +
                                      cl_sync_1x1_bb[None, None, :] * f_sync_1x1)
    if add_02:
        # 0x2
        f_dust_0x2 = np.outer(f_dust, f_dust_2)
        f_dust_0x2 = (0.5*(f_dust_0x2 + f_dust_0x2.T))[:, :, None]
        f_sync_0x2 = np.outer(f_sync, f_sync_2)
        f_sync_0x2 = (0.5*(f_sync_0x2 + f_sync_0x2.T))[:, :, None]
        C_ells_sky[:, 0, :, 0, :] += (cl_dust_0x2_ee[None, None, :] * f_dust_0x2 +
                                      cl_sync_0x2_ee[None, None, :] * f_sync_0x2)
        C_ells_sky[:, 1, :, 1, :] += (cl_dust_0x2_bb[None, None, :] * f_dust_0x2 +
                                      cl_sync_0x2_bb[None, None, :] * f_sync_0x2)

    # to D_ell
    C_ells_sky *= cl2dl[None, None, None, None, :]
    
    l_binned, windows, cls_binned = bin_cls(C_ells_sky,
                                            delta_ell=delta_ell)
    indices, cls_binned, cov_binned = get_vector_and_covar(l_binned,
                                                           cls_binned)
    dict_out = {'ls_unbinned': ells,
                'cls_unbinned': C_ells_sky,
                'ls_binned': l_binned,
                'cls_binned': cls_binned,
                'cov_binned': cov_binned,
                'ind_cl': indices,
                'windows': windows}
    return dict_out

def get_sky_realization(nside, seed=None, mean_pars=None, moment_pars=None,
                        compute_cls=False, delta_ell=10):
    """ Generate a sky realization for a set of input sky parameters.

    Args:
        nside: HEALPix resolution parameter.
        seed: seed to be used (if `None`, then a random seed will
            be used).
        mean_pars: mean parameters (see `get_default_params`).
            If `None`, then a default set will be used.
        moment_pars: mean parameters (see `get_default_params`).
            If `None`, then a default set will be used.
        compute_cls: return also the power spectra? Default: False.
        delta_ell: bandpower size to use if compute_cls is True.

    Returns:
        A dictionary containing the different component maps,
        spectral index maps and frequency maps.
        If `compute_cls=True`, then the dictionary will also
        contain information of the signal, noise and total 
        (i.e. signal + noise) power spectra. 
    """
    nu = get_freqs()
    npix = hp.nside2npix(nside)
    if seed is not None:
        np.random.seed(seed)
    if mean_pars is None:
        mean_pars, _ = get_default_params()
    if moment_pars is None:
        _, moment_pars = get_default_params()
    lmax = 3*nside-1
    ells, dl2cl, cl2dl, cl_dust_bb, cl_dust_ee, cl_sync_bb, cl_sync_ee, cl_cmb_bb, cl_cmb_ee = get_mean_spectra(lmax, mean_pars)
    cl0 = 0 * cl_dust_bb
    # Dust amplitudes
    Q_dust, U_dust = hp.synfast([cl0, cl_dust_ee, cl_dust_bb, cl0, cl0, cl0],
                                nside, new=True, verbose=False)[1:]
    # Sync amplitudes
    Q_sync, U_sync = hp.synfast([cl0, cl_sync_ee, cl_sync_bb, cl0, cl0, cl0],
                                nside, new=True, verbose=False)[1:]

    # CMB amplitude
    Q_cmb, U_cmb = hp.synfast([cl0, cl_cmb_ee, cl_cmb_bb, cl0, cl0, cl0],
                              nside, new=True, verbose=False)[1:]

    # Dust spectral index
    beta_dust = get_beta_map(nside,
                             mean_pars['beta_dust'],
                             moment_pars['amp_beta_dust'],
                             moment_pars['gamma_beta_dust'],
                             moment_pars['l0_beta_dust'],
                             moment_pars['l_cutoff_beta_dust'])
    # Dust temperature
    temp_dust = np.ones(npix) * mean_pars['temp_dust']
    # Synchrotron spectral index
    beta_sync = get_beta_map(nside, 
                             mean_pars['beta_sync'],
                             moment_pars['amp_beta_sync'],
                             moment_pars['gamma_beta_sync'],
                             moment_pars['l0_beta_sync'],
                             moment_pars['l_cutoff_beta_sync'])

    # Create PySM simulation
    zeromap = np.zeros(npix)
    # Dust
    d2 = models("d2", nside)
    d2[0]['nu_0_I'] = mean_pars['nu0_dust']
    d2[0]['nu_0_P'] = mean_pars['nu0_dust']
    d2[0]['A_I'] = zeromap
    d2[0]['A_Q'] = Q_dust
    d2[0]['A_U'] = U_dust
    d2[0]['spectral_index'] = beta_dust
    d2[0]['temp'] = temp_dust
    # Sync
    s1 = models("s1", nside)
    s1[0]['nu_0_I'] = mean_pars['nu0_sync']
    s1[0]['nu_0_P'] = mean_pars['nu0_sync']
    s1[0]['A_I'] = zeromap
    s1[0]['A_Q'] = Q_sync
    s1[0]['A_U'] = U_sync
    s1[0]['spectral_index'] = beta_sync
    # CMB
    c1 = models("c1", nside)
    c1[0]['model'] = 'pre_computed' #different output maps at different seeds 
    c1[0]['A_I'] = zeromap
    c1[0]['A_Q'] = Q_cmb
    c1[0]['A_U'] = U_cmb

    sky_config = {'dust' : d2, 'synchrotron' : s1, 'cmb' : c1}
    sky = pysm.Sky(sky_config)
    instrument_config = {
        'nside' : nside,
        'frequencies' : nu, #Expected in GHz 
        'use_smoothing' : False,
        'beams' : np.ones_like(nu), #Expected in arcmin 
        'add_noise' : False,
        'use_bandpass' : False,
        'channel_names' : ['LF1', 'LF2', 'MF1', 'MF2', 'UHF1', 'UHF2'],
        'output_units' : 'uK_RJ',
        'output_directory' : 'none',
        'output_prefix' : 'none',
    }
    sky = pysm.Sky(sky_config)
    instrument = pysm.Instrument(instrument_config)
    maps_signal, _ = instrument.observe(sky, write_outputs=False)
    maps_signal = maps_signal[:,1:,:]
    # Change to CMB units
    maps_signal = maps_signal/fcmb(nu)[:,None,None]

    dict_out = {'maps_dust': np.array([Q_dust, U_dust]),
                'maps_sync': np.array([Q_sync, U_sync]),
                'maps_cmb': np.array([Q_cmb, U_cmb]),
                'beta_dust': beta_dust,
                'beta_sync': beta_sync,
                'freq_maps': maps_signal}

    if compute_cls:
        cls_unbinned = map2cl(maps_signal)
        l_binned, windows, cls_binned = bin_cls(cls_unbinned,
                                                delta_ell=delta_ell)
        indices, cls_binned, cov_binned = get_vector_and_covar(l_binned,
                                                               cls_binned)
        dict_out['ls_unbinned'] = ells
        dict_out['cls_unbinned'] = cls_unbinned
        dict_out['ls_binned'] = l_binned
        dict_out['cls_binned'] = cls_binned
        dict_out['cov_binned'] = cov_binned
        dict_out['ind_cl'] = indices
        dict_out['windows'] = windows

    return dict_out

def create_noise_splits(nside, add_mask=False, sens=1, knee=1, ylf=1,
                        fsky=0.1, nsplits=4):
    """ Generate instrumental noise realizations.

    Args:
        nside: HEALPix resolution parameter.
        seed: seed to be used (if `None`, then a random seed will
            be used).
        add_mask: return the masked splits? Default: False. 
        sens: sensitivity (0, 1 or 2)
        knee: knee type (0 or 1)
        ylf: number of years for the LF tube.
        fsky: sky fraction to use for the noise realizations.
        nsplits: number of splits (i.e. independent noise realizations).

    Returns:
        A dictionary containing the noise maps.
        If `add_mask=True`, then the masked noise maps will
        be returned.
    """
    nu = get_freqs()
    nfreq = len(nu)
    lmax = 3*nside-1
    ells = np.arange(lmax+1)
    nells = len(ells)
    dl2cl = np.ones(len(ells))
    dl2cl[1:] = 2*np.pi/(ells[1:]*(ells[1:]+1.))
    cl2dl = (ells*(ells+1.))/(2*np.pi)
    nell=np.zeros([nfreq,lmax+1])
    _,nell[:,2:],_=Simons_Observatory_V3_SA_noise(sens,knee,ylf,fsky,lmax+1,1)
    nell*=cl2dl[None,:]

    npol = 2
    nmaps = nfreq*npol
    N_ells = np.zeros([nfreq, npol, nfreq, npol, nells])
    for i,n in enumerate(nu):
        for j in [0,1]:
            N_ells[i, j, i, j, :] = nell[i]
        
    # Noise maps
    npix = hp.nside2npix(nside)
    maps_noise = np.zeros([nsplits, nfreq, npol, npix])
    for s in range(nsplits):
        for i in range(nfreq):
            nell_ee = N_ells[i, 0, i, 0, :]*dl2cl * nsplits
            nell_bb = N_ells[i, 1, i, 1, :]*dl2cl * nsplits
            nell_00 = nell_ee * 0 * nsplits
            maps_noise[s, i, :, :] = hp.synfast([nell_00, nell_ee, nell_bb,
                                                 nell_00, nell_00, nell_00],
                                                nside, pol=False, new=True,
                                                verbose=False)[1:]

    if add_mask:
        nhits=hp.ud_grade(hp.read_map("norm_nHits_SA_35FOV.fits",  verbose=False),nside_out=nside)
        nhits/=np.amax(nhits) 
        fsky_msk=np.mean(nhits) 
        nhits_binary=np.zeros_like(nhits) 
        nhits_binary[nhits>1E-3]=1

    dict_out = {'maps_noise': maps_noise, 'cls_noise': N_ells}
    return dict_out
