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

dirname = "simulations_only_sync/sims_ns%d_seed%d_lm%d"%(o.nside, o.seed, o.ellmax)
os.system('mkdir -p '+dirname)
print(dirname)

# No spectral index variations
mean_p, moment_p = ut.get_default_params()
mean_p['include_CMB'] = False
mean_p['include_sync'] = True
mean_p['include_dust'] = False
#print(moment_p['amp_beta_dust'], moment_p['amp_beta_sync'])
sim_db0p0 = ut.get_sky_realization(o.nside, seed=o.seed, mean_pars=mean_p,
                                   moment_pars=moment_p, compute_cls=True)
thr_db0p0 = ut.get_theory_spectra(o.nside, mean_pars=mean_p,
                                  moment_pars=moment_p, add_11=True, add_02=True)
mps_signal_db0p0 = sim_db0p0['freq_maps']
obs_splits_db0p0 = ut.create_noise_splits(o.nside, maps_signal=mps_signal_db0p0,
                                          seed=o.seed, compute_cls=True,
                                          mean_pars=mean_p, moment_pars=moment_p)

# Spectral index variantions for dust with std 0.2
amp_beta_dust = ut.get_delta_beta_amp(sigma=0.2, gamma=-3.)
#print(amp_beta_dust)
moment_p['amp_beta_dust'] = amp_beta_dust
sim_db0p2 = ut.get_sky_realization(o.nside, seed=o.seed, mean_pars=mean_p,
                                   moment_pars=moment_p, compute_cls=True)
thr_db0p2 = ut.get_theory_spectra(o.nside, mean_pars=mean_p,
                                  moment_pars=moment_p, add_11=True, add_02=True)
mps_signal_db0p2 = sim_db0p2['freq_maps']
obs_splits_db0p2 = ut.create_noise_splits(o.nside, maps_signal=mps_signal_db0p2,
                                          seed=o.seed, compute_cls=True,
                                          mean_pars=mean_p, moment_pars=moment_p)
for i_nu1 in range(6):
    for i_nu2 in range(i_nu1, 6):
        i_pol = 1  # only consider B
        i_map1 = i_pol + 2*i_nu1
        i_map2 = i_pol + 2*i_nu2
        index_x = obs_splits_db0p2['ind_cl'][i_map1, i_map2]

        # Simulation b0p0
        cl_sig_b0p0 = obs_splits_db0p0['cls_binned_signal'][index_x, :]
        cov_sig_b0p0 = obs_splits_db0p0['cov_binned_signal'][index_x, :, index_x, :]
        err_sig = np.sqrt(np.diag(cov_sig_b0p0))
        cl_noi_b0p0 = obs_splits_db0p0['cls_binned_noise'][index_x, :]
        cov_noi_b0p0 = obs_splits_db0p0['cov_binned_noise'][index_x, :, index_x, :]
        err_noi = np.sqrt(np.diag(cov_noi_b0p0))
        cl_tot_b0p0 = obs_splits_db0p0['cls_binned_total'][index_x, :]
        cov_tot_b0p0 = obs_splits_db0p0['cov_binned_total'][index_x, :, index_x, :]
        err_tot = np.sqrt(np.diag(cov_tot_b0p0))

        # Simulation b0p2
        cl_sig_b0p2 = obs_splits_db0p2['cls_binned_signal'][index_x, :]
        cl_noi_b0p2 = obs_splits_db0p2['cls_binned_noise'][index_x, :]
        cl_tot_b0p2 = obs_splits_db0p2['cls_binned_total'][index_x, :]
        
        # Theory power spectra
        cl_t_b0p0 = thr_db0p0['cls_binned'][index_x, :]
        cl_t_b0p2 = thr_db0p2['cls_binned'][index_x, :]

        l = sim_db0p2['ls_binned']
        mask = l <= 2*o.nside

        # Save to file
        np.savetxt(dirname+'/ells.txt', l[mask])        
        np.savetxt(dirname+'/cl_d_b0p0_%d_%d_sig.txt'% (i_nu1, i_nu2),  cl_sig_b0p0[mask])
        np.savetxt(dirname+'/cl_d_b0p0_%d_%d_noi.txt'% (i_nu1, i_nu2),  cl_noi_b0p0[mask])
        np.savetxt(dirname+'/cl_d_b0p0_%d_%d_tot.txt'% (i_nu1, i_nu2),  cl_tot_b0p0[mask])        
        np.savetxt(dirname+'/cl_d_b0p2_%d_%d_sig.txt'% (i_nu1, i_nu2),  cl_sig_b0p2[mask])
        np.savetxt(dirname+'/cl_d_b0p2_%d_%d_noi.txt'% (i_nu1, i_nu2),  cl_noi_b0p2[mask])
        np.savetxt(dirname+'/cl_d_b0p2_%d_%d_tot.txt'% (i_nu1, i_nu2),  cl_tot_b0p2[mask])
        np.savetxt(dirname+'/cl_t_b0p0_%d_%d.txt'% (i_nu1, i_nu2),  cl_t_b0p0[mask])
        np.savetxt(dirname+'/cl_t_b0p2_%d_%d.txt'% (i_nu1, i_nu2),  cl_t_b0p2[mask])        
        np.savetxt(dirname+'/err_%d_%d_sig.txt'% (i_nu1, i_nu2), err_sig[mask])
        np.savetxt(dirname+'/err_%d_%d_noi.txt'% (i_nu1, i_nu2), err_noi[mask])
        np.savetxt(dirname+'/err_%d_%d_tot.txt'% (i_nu1, i_nu2), err_tot[mask])






