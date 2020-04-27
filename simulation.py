import utils as ut
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import os
from optparse import OptionParser

parser = OptionParser()
parser.add_option('--seed', dest='seed',  default=1000, type=int,
                  help='Set to define seed, default=1000')
parser.add_option('--nside', dest='nside', default=256, type=int,
                  help='Set to define Nside parameter, default=256')
parser.add_option('--std', dest='std', default=0., type=float,
                  help='')
parser.add_option('--include-cmb', dest='include_cmb', default=True, action='store_true',
                  help='Set to include CMB to simulation, default=True.')
parser.add_option('--include-sync', dest='include_sync', default=True, action='store_true',
                  help='Set to include synchrotron to simulation, default=True.')
parser.add_option('--include-dust', dest='include_dust', default=True, action='store_true',
                  help='Set to include dust to simulation, default=True.')
parser.add_option('--include-E', dest='include_E', default=False, action='store_true',
                  help='Set to include E-modes to simulation, default=False.')
parser.add_option('--include-B', dest='include_B', default=True, action='store_true',
                  help='Set to include B-modes to simulation, default=True.')
(o, args) = parser.parse_args()

nside = o.nside
seed = o.seed

dirname = "/mnt/extraspace/susanna/Simulations_Moments/sim_ns%d_seed%d_std%d"%(o.nside, o.seed, o.std*100)
os.system('mkdir -p '+dirname)
print(dirname)

# Decide whether spectral index is constant or varying
mean_p, moment_p = ut.get_default_params()
if o.std != 0. :
    # Spectral index variantions for dust with std
    amp_beta_dust = ut.get_delta_beta_amp(sigma=o.std, gamma=-3.)
    moment_p['amp_beta_dust'] = amp_beta_dust

# Define parameters for the simulation:
# Which components do we want to include?
mean_p['include_CMB'] = o.include_cmb
mean_p['include_sync'] = o.include_sync
mean_p['include_dust'] = o.include_dust
# Which polarizations do we want to include?
mean_p['include_E'] = o.include_E
mean_p['include_B'] = o.include_B

# Theory prediction, simulation and noise
thr = ut.get_theory_spectra(o.nside, mean_pars=mean_p,
                                  moment_pars=moment_p, add_11=True, add_02=True)

sim = ut.get_sky_realization(o.nside, seed=o.seed, mean_pars=mean_p,
                                   moment_pars=moment_p, compute_cls=True)

noi = ut.create_noise_splits(o.nside)

# Define maps signal and noise
mps_signal = sim['freq_maps']
mps_noise = noi['maps_noise']

# Save sky maps
nu = ut.get_freqs()
nfreq = len(nu)
npol = 2
nmaps = nfreq*npol
npix = hp.nside2npix(o.nside)
hp.write_map(dirname+"/maps_sky_signal.fits", mps_signal.reshape([nmaps,npix]),
             overwrite=True)

# Create splits
nsplits = 4
for s in range(nsplits):
    hp.write_map(dirname+"/obs_split%dof%d.fits.gz" % (s+1, nsplits),
                 (mps_signal[:,:,:]+mps_noise[s,:,:,:]).reshape([nmaps,npix]),
                 overwrite=True)

# Write splits list
f=open(dirname+"/splits_list.txt","w")
Xout=""
for i in range(nsplits):
    Xout += dirname+'/obs_split%dof%d.fits.gz\n' % (i+1, nsplits)
f.write(Xout)
f.close()





