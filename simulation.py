import utils as ut
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import os
from optparse import OptionParser

parser = OptionParser()
parser.add_option('--lmax', dest='ellmax',  default=383, type=int,
                  help='Set to define lmax for w3j, default=383')
parser.add_option('--seed', dest='seed',  default=1000, type=int,
                  help='Set to define seed, default=1000')
parser.add_option('--nside', dest='nside', default=256, type=int,
                  help='Set to define Nside parameter, default=256')
(o, args) = parser.parse_args()

nside = o.nside
seed = o.seed

dirname = "simulations_only_dust/sims_ns%d_seed%d_lm%d"%(o.nside, o.seed, o.ellmax)
os.system('mkdir -p '+dirname)
print(dirname)

# First, let's generate a set of parameters
mean_p, moment_p = ut.get_default_params()
# Note that you can remove individual components by doing e.g.:
mean_p['include_CMB'] = False
mean_p['include_sync'] = False
mean_p['include_dust'] = True
# By default, these won't contain any spectral index variations:
print(moment_p['amp_beta_dust'], moment_p['amp_beta_sync'])

# OK, let's look at data with no spectral index variations.
#
# First let's generate a simulation with these parameters
sim_db0p0 = ut.get_sky_realization(o.nside, seed=o.seed, mean_pars=mean_p,
                                   moment_pars=moment_p, compute_cls=True)
# And now, let's compute their associated theory prediction
thr_db0p0 = ut.get_theory_spectra(o.nside, mean_pars=mean_p,
                                  moment_pars=moment_p)

# Now, let's introduce spectral index variantions for dust
# with a standard deviation of 0.2.
# First, we can get the corresponding power spectrum amplitude with
# this little function:
amp_beta_dust = ut.get_delta_beta_amp(sigma=0.2, gamma=-3.)
print(amp_beta_dust)
# Now let's modify the moment parameters to introduce spatial variations:
moment_p['amp_beta_dust'] = amp_beta_dust
# And now let's generate a simulation and the theory prediction:
sim_db0p2 = ut.get_sky_realization(o.nside, seed=o.seed, mean_pars=mean_p,
                                   moment_pars=moment_p, compute_cls=True)
thr_db0p2 = ut.get_theory_spectra(o.nside, mean_pars=mean_p,
                                  moment_pars=moment_p)

# OK, now let's compare everything!
# We will generate one plot for each pair of frequencies,
# and we will only look at BB
for i_nu1 in range(6):  # Loop over frequencies
    for i_nu2 in range(i_nu1, 6):  # Loop over frequencies
        # The routines above return power spectra in a vectorized form
        # (i.e. with all cross-correlations in a single dimension).
        # To transform between cross-correlation indices and map indices
        # we can use the following:
        i_pol = 1  # 0 for E, 1 for B
        # Map indices
        i_map1 = i_pol + 2*i_nu1
        i_map2 = i_pol + 2*i_nu2
        # Cross-correlation index
        index_x = sim_db0p0['ind_cl'][i_map1, i_map2]
        # OK, now let's select the power spectra
        # Data
        cl_d_b0p0 = sim_db0p0['cls_binned'][index_x, :]
        cl_d_b0p2 = sim_db0p2['cls_binned'][index_x, :]
        # And theory
        cl_t_b0p0 = thr_db0p0['cls_binned'][index_x, :]
        cl_t_b0p2 = thr_db0p2['cls_binned'][index_x, :]
        # As well as their errorbar
        cov_b0p0 = sim_db0p0['cov_binned'][index_x, :, index_x, :]
        err_b0p0 = np.sqrt(np.diag(cov_b0p0))
        cov_b0p2 = sim_db0p2['cov_binned'][index_x, :, index_x, :]
        err_b0p2 = np.sqrt(np.diag(cov_b0p2))
        # OK, plotting time!
        l = sim_db0p0['ls_binned']
        # It's not very accurate to plot anything beyond 2*nside, so we'll remove that
        mask = l <= 2*o.nside

        np.savetxt(dirname+'/ells.txt', l[mask])        
        np.savetxt(dirname+'/cl_d_b0p0_%d_%d.txt'% (i_nu1, i_nu2),  cl_d_b0p0[mask])
        np.savetxt(dirname+'/cl_t_b0p0_%d_%d.txt'% (i_nu1, i_nu2),  cl_t_b0p0[mask])
        np.savetxt(dirname+'/cl_d_b0p2_%d_%d.txt'% (i_nu1, i_nu2),  cl_d_b0p2[mask])
        np.savetxt(dirname+'/cl_t_b0p2_%d_%d.txt'% (i_nu1, i_nu2),  cl_t_b0p2[mask])
        np.savetxt(dirname+'/err_b0p0_%d_%d.txt'% (i_nu1, i_nu2), err_b0p0[mask])
        np.savetxt(dirname+'/err_b0p2_%d_%d.txt'% (i_nu1, i_nu2), err_b0p2[mask])

