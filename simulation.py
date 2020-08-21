import utils as ut
import healpy as hp
import numpy as np
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
                  help='Deviation from the mean value of beta dust, default = 0.')
parser.add_option('--std-sync', dest='std_sync', default=0., type=float,
                  help='Deviation from the mean value of beta synchrotron, default = 0.')
parser.add_option('--gamma-dust', dest='gamma_dust', default=-3., type=float,
                  help='Exponent of the beta dust power law, default=-3.')
parser.add_option('--gamma-sync', dest='gamma_sync', default=-3., type=float,
                  help='Exponent of the beta sync power law, default=-3.')
parser.add_option('--include-cmb', dest='include_cmb', default=True, action='store_false',
                  help='Set to remove CMB from simulation, default=True.')
parser.add_option('--include-sync', dest='include_sync', default=True, action='store_false',
                  help='Set to remove synchrotron from simulation, default=True.')
parser.add_option('--include-dust', dest='include_dust', default=True, action='store_false',
                  help='Set to remove dust from simulation, default=True.')
parser.add_option('--include-E', dest='include_E', default=True, action='store_false',
                  help='Set to remove E-modes from simulation, default=True.')
parser.add_option('--include-B', dest='include_B', default=True, action='store_false',
                  help='Set to remove B-modes from simulation, default=True.')
parser.add_option('--mask', dest='add_mask', default=False, action='store_true',
                  help='Set to add mask to observational splits, default=False.')
parser.add_option('--dust-vansyngel', dest='dust_vansyngel', default=False, action='store_true',
                  help='Set to use Vansyngel et al\'s dust model, default=False.')
parser.add_option('--beta', dest='gaussian_beta', default=True, action='store_false',
                  help='Set for non-gaussian beta variation.')
parser.add_option('--nu0-dust', dest='nu0_dust', default=353., type=int,
                  help='Set to change dust pivot frequency, default=353 GHz.')
parser.add_option('--nu0-sync', dest='nu0_sync', default=23., type=int,
                  help='Set to change synchrotron pivot frequency, default=23 GHz.')
parser.add_option('--A-dust-BB', dest='Ad', default=5, type=float,
                  help='Set to modify the B-mode dust power spectrum amplitude, default=5')
parser.add_option('--alpha-dust-BB', dest='alpha_d', default=-0.42, type=float,
                  help='Set to mofify tilt in D_l^BB for dust, default=-0.42')
parser.add_option('--A-sync-BB', dest='As', default=2, type=float,
                  help='Set to modify the B-mode dust power spectrum amplitude, default=2')
parser.add_option('--alpha-sync-BB', dest='alpha_s', default=-0.6, type=float,
                  help='Set to mofify tilt in D_l^BB for synchrotron, default=-0.42')
parser.add_option('--plaw-amp', dest='plaw_amps', default=True, action='store_false',
                  help='Set to use realistic amplitude maps for dust and synchrotron.')
parser.add_option('--r-tensor', dest='r_tensor', default=0.0, type=float,
                  help='Set to mofify tensor-to-scalar ratio')
parser.add_option('--dust-sed', dest='dust_sed', default='mbb', type=str,
                  help='Dust SED (\'mbb\' or \'hensley_draine\')')
(o, args) = parser.parse_args()

nside = o.nside
seed = o.seed

if o.dirname == 'none':
    #o.dirname = "/mnt/extraspace/susanna/BBMoments/Simulations_Moments/sim_ns%d" % o.nside
    #o.dirname = "/mnt/extraspace/susanna/BBMoments/Simulations_Moments_shiftednu0s_try2/sim_ns%d" % o.nside
    #o.dirname = "/mnt/extraspace/susanna/BBMoments/Simulations_Moments_realisticPySMsims_updated/sim_ns%d" % o.nside
    #####o.dirname = "/mnt/extraspace/susanna/BBMoments/Simulations_Moments_realAmpsBetas_FITVALUES/sim_ns%d" % o.nside
    #o.dirname = "/mnt/zfsusers/susanna/PySM-tests2/Nmt_compare_Cls/FITS_sim_ns%d" % o.nside
    o.dirname = "./FITS2_sim_ns%d" % o.nside
    o.dirname+= "_seed%d" % o.seed
    o.dirname+= "_stdd%d_stds%d"%(o.std_dust*100, o.std_sync*100)
    o.dirname+= "_gdm%.1lf_gsm%.1lf"%(-int(o.gamma_dust), -int(o.gamma_sync))
    if o.add_mask:
        o.dirname+= "_msk"
    else:
        o.dirname+= "_fullsky"
    if o.include_E:
        o.dirname+= "_E"
    if o.include_B:
        o.dirname+= "_B"
    if not o.gaussian_beta:
        o.dirname+= "_pysmBetas"
    if not o.plaw_amps:
        o.dirname+= "_realAmps"
    o.dirname+= "_nu0d%d_nu0s%d" %(o.nu0_dust, o.nu0_sync)
    if o.include_cmb:
        o.dirname+= "_cmb"
    if o.include_dust:
        o.dirname+= "_dust"
    if o.include_sync:
        o.dirname+= "_sync"
    o.dirname+= "_Ad%d" %(o.Ad)
os.system('mkdir -p ' + o.dirname)
print(o.dirname)

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
if o.dust_vansyngel:
    mean_p['include_dust'] = False
else:
    mean_p['include_dust'] = o.include_dust

# Which polarizations do we want to include?
mean_p['include_E'] = o.include_E
mean_p['include_B'] = o.include_B

# Modify r
mean_p['r_tensor'] = o.r_tensor

# Dust SED
mean_p['dust_SED'] = o.dust_sed

# Modify template plaws for dust and sync
# i.e. define amp_d_bb, amp_s_bb, alpha_s, alpha_d
mean_p['A_dust_BB'] = o.Ad
mean_p['alpha_dust_BB'] = o.alpha_d
mean_p['A_sync_BB'] = o.As
mean_p['alpha_sync_BB'] = o.alpha_s

# Define pivot freqs
mean_p['nu0_dust'] = o.nu0_dust
mean_p['nu0_sync'] = o.nu0_sync

# Theory prediction, simulation and noise
thr = ut.get_theory_spectra(o.nside, mean_pars=mean_p,
                            moment_pars=moment_p, add_11=True, add_02=True)
scc = ut.get_theory_sacc(o.nside, mean_pars=mean_p,
                         moment_pars=moment_p, add_11=True, add_02=True)
scc.saveToHDF(o.dirname+"/cells_model.sacc")
sim = ut.get_sky_realization(o.nside, seed=o.seed, mean_pars=mean_p,
                             moment_pars=moment_p, gaussian_betas=o.gaussian_beta,
                             plaw_amps=o.plaw_amps, compute_cls=True)
if o.dust_vansyngel:
    import utils_vansyngel as uv
    nus = ut.get_freqs()
    qd, ud = uv.get_dust_sim(nus, o.nside)
    print(sim['freq_maps'].shape)
    print(qd.shape)
    exit(1)
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

###Added
mps_dust_amp = sim['maps_dust']
mps_sync_amp = sim['maps_sync']
hp.write_map(o.dirname+"/maps_dust_QU.fits", mps_dust_amp,
             overwrite=True)
hp.write_map(o.dirname+"/maps_sync_QU.fits", mps_sync_amp,
             overwrite=True)

# Create splits
nsplits = len(mps_noise)
for s in range(nsplits):
    maps_signoi = mps_signal[:,:,:]+mps_noise[s,:,:,:]
    if o.add_mask:
        maps_signoi *= noi['mask']
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
