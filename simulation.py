import utils as ut
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import os
from optparse import OptionParser

parser = OptionParser()
parser.add_option('--output-dir', dest='dirname', default='none',
                  type=str, help='Output directory')
parser.add_option('--seed', dest='seed',  default=1000, type=int,
                  help='Set to define seed, default=1000')
parser.add_option('--nside', dest='nside', default=256, type=int,
                  help='Set to define Nside parameter, default=256')
parser.add_option('--std-dust', dest='std_dust', default=0., type=float,
                  help='')
parser.add_option('--std-sync', dest='std_sync', default=0., type=float,
                  help='')
parser.add_option('--gamma-dust', dest='gamma_dust', default=-3., type=float,
                  help='')
parser.add_option('--gamma-sync', dest='gamma_sync', default=-3., type=float,
                  help='')
parser.add_option('--remove-cmb', dest='include_cmb', default=True, action='store_false',
                  help='Set to remove CMB from simulation, default=True.')
parser.add_option('--remove-sync', dest='include_sync', default=True, action='store_false',
                  help='Set to remove synchrotron from simulation, default=True.')
parser.add_option('--remove-dust', dest='include_dust', default=True, action='store_false',
                  help='Set to remove dust from simulation, default=True.')
parser.add_option('--remove-E', dest='include_E', default=True, action='store_false',
                  help='Set to remove E-modes from simulation, default=True.')
parser.add_option('--remove-B', dest='include_B', default=True, action='store_false',
                  help='Set to remove B-modes from simulation, default=True.')
parser.add_option('--mask', dest='add_mask', default=False, action='store_true',
                  help='Set to add mask to observational splits.')
(o, args) = parser.parse_args()

nside = o.nside
seed = o.seed

if o.dirname == 'none':
    o.dirname = "/mnt/extraspace/damonge/Simulations_Moments/sim_ns%d" % o.nside
    o.dirname+= "_seed%d" % o.seed
    o.dirname+= "_stdd%d_stds%d"%(o.std_dust*100, o.std_sync*100)
    o.dirname+= "_gdm%.1lf_gsm%.1lf"%(-o.gamma_dust, -o.gamma_sync)
os.system('mkdir -p ' + o.dirname)
print(dirname)

# Decide whether spectral index is constant or varying
mean_p, moment_p = ut.get_default_params()
if o.std_dust > 0. :
    # Spectral index variantions for dust with std
    amp_beta_dust = ut.get_delta_beta_amp(sigma=o.std_dust, gamma=o.gamma_dust)
    moment_p['amp_beta_dust'] = amp_beta_dust
    moment_p['gamma_beta_dust'] = o.gamma_dust
if o.std_sync > 0. :
    # Spectral index variantions for sync with std
    amp_beta_sync = ut.get_delta_beta_amp(sigma=o.std_sync, gamma=o.gamma_sync)
    moment_p['amp_beta_sync'] = amp_beta_sync
    moment_p['gamma_beta_sync'] = o.gamma_sync

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
scc = ut.get_theory_sacc(o.nside, mean_pars=mean_p,
                         moment_pars=moment_p, add_11=True, add_02=True)
scc.saveToHDF(o.dirname+"/cells_model.sacc")
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
hp.write_map(o.dirname+"/maps_sky_signal.fits", mps_signal.reshape([nmaps,npix]),
             overwrite=True)

# Create splits
nsplits = len(mps_noise)
for s in range(nsplits):
    maps_signoi = mps_signal[:,:,:]+mps_noise[s,:,:,:]
    if o.add_mask:
        maps_signoi *= noi['nhits']
    hp.write_map(o.dirname+"/obs_split%dof%d.fits.gz" % (s+1, nsplits),
                 (maps_signoi).reshape([nmaps,npix]),
                 overwrite=True)

# Write splits list
f=open(o.dirname+"/splits_list.txt","w")
Xout=""
for i in range(nsplits):
    Xout += o.dirname+'/obs_split%dof%d.fits.gz\n' % (i+1, nsplits)
f.write(Xout)
f.close()
