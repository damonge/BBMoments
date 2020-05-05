import utils as ut
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import os
from optparse import OptionParser

parser = OptionParser()
parser.add_option('--data-dir', dest='ddirname', default='none',
                  type=str, help='Data directory')
parser.add_option('--output-dir', dest='odirname', default='none',
                  type=str, help='Output directory')
parser.add_option('--nside', dest='nside', default=256, type=int,
                  help='Set to define Nside parameter, default=256')
(o, args) = parser.parse_args()

nside = o.nside

if o.odirname == 'none':
    o.odirname = "bbpipe_out"
os.system('mkdir -p ' + o.odirname)

def get_ini(oo):
    yml = {'modules': 'bbpower'}
    yml['launcher'] = 'local'
    yml['stages'] = [{'name': 'BBPowerSpecter',
                      'nprocess': 1},
                     {'name': 'BBPowerSummarizer',
                      'nprocess': 1},
                     {'name': 'BBCompSep',
                      'nprocess': 1},
                     {'name': 'BBPlotter',
                      'nprocess': 1}]
    yml['inputs'] = {'splits_list': oo.ddirname+'/splits_list.txt',
                     'bandpasses_list': oo.ddirname+'/bandpasses.txt',
                     'beams_list': oo.ddirname+'/beams.txt',
                     'masks_apodized': oo.ddirname+'/mask.fits',
                     'sims_list': oo.ddirname+'/sims_list.txt',
                     'cells_fiducial': oo.ddirname+'/cells_model.sacc'}
    yml['config'] = oo.odirname+'/config.yaml'
    yml['resume'] = False
    yml['output_dir'] = oo.odirname
    yml['log_dir'] = oo.odirname
    yml['pipeline_log'] = oo.odirname+'/log.txt'
    return yml


def get_config(oo):
    yml = {'global':
           {'nside': oo.nside,
            'compute_dell': True}}
    yml['BBPowerSpecter']={'bpw_edges': oo.ddirname+"/bpw_edges.txt",
                           'purify_B': False,
                           'n_iter' : 3}
    yml['BBPowerSummarizer']={'nulls_covar_type': "diagonal",
                              'nulls_covar_diag_order': 0,
                              'data_covar_type': "block_diagonal",
                              "data_covar_diag_order": 3}

    d = {'sampler': 'emcee',
         'nwalkers': 128,
         'n_iters': 1000,
         'likelihood_type': 'chi2',
         'pol_channels': ['B'],
         'l_min': 30,
         'l_max': 300}
    d['cmb_model'] = {'cmb_templates': ["./examples/data/camb_lens_nobb.dat", 
                                        "./examples/data/camb_lens_r1.dat"],
                      'params': {'r_tensor': ['r_tensor', 'tophat', [-1, 0.00, 1]],
                                 'A_lens': ['A_lens', 'tophat', [0.00,1.0,2.00]]}
                      }
    fg1 ={'name': 'Dust',
          'sed': 'Dust',
          'cl':{'EE': 'ClPowerLaw',
                'BB': 'ClPowerLaw'},
          'sed_parameters':{'beta_d': ['beta_d', 'Gaussian', [1.6, 0.11]],
                            'temp_d': ['temp', 'fixed', [19.6]],
                            'nu0_d': ['nu0', 'fixed', [353.]]},
          'cl_parameters':{'EE':
                           {'amp_d_ee': ['amp', 'tophat', [0., 10., "inf"]],
                            'alpha_d_ee': ['alpha', 'tophat', [-1., -0.42, 0.]],
                            'l0_d_ee': ['ell0', 'fixed', [80.]]},
                           'BB':
                           {'amp_d_bb': ['amp', 'tophat', [0., 5., "inf"]],
                            'alpha_d_bb': ['alpha', 'tophat', [-1., -0.42, 0.]],
                            'l0_d_bb': ['ell0', 'fixed', [80.]]}},
          'cross':{'epsilon_ds': ['component_2', 'tophat', [-1., 0., 1.]]}}
    #'component_2':
    fg2 = {'name': 'Synchrotron',
           'sed': 'Synchrotron',
           'cl':{'EE': 'ClPowerLaw',
                 'BB': 'ClPowerLaw'},
           'sed_parameters': {'beta_s': ['beta_pl', 'Gaussian', [-3.0, 0.3]],
                              'nu0_s': ['nu0', 'fixed', [23.]]},
           'cl_parameters': {'EE':
                             {'amp_s_ee': ['amp', 'tophat', [0., 4., "inf"]],
                              'alpha_s_ee': ['alpha', 'tophat', [-1., -0.6, 0.]],
                              'l0_s_ee': ['ell0', 'fixed', [80.]]},
                             'BB':
                             {'amp_s_bb': ['amp', 'tophat', [0., 2., "inf"]],
                              'alpha_s_bb': ['alpha', 'tophat', [-1., -0.6, 0.]],
                              'l0_s_bb': ['ell0', 'fixed', [80.]]}}}
    fg = {'component_1': fg1,
          'component_2': fg2}
    d['fg_model'] = fg

    yml['BBCompSep'] = d
    return yml

def get_bandpasses(oo):
    freqs = ut.get_freqs()
    stout = ''
    for i_f, f in enumerate(freqs):
        fs = np.array([f-1, f, f+1])
        bs = np.array([0., 1., 0.])
        fname = oo.ddirname+'/bpss_%d.txt' % i_f
        stout += fname + '\n'
        np.savetxt(fname, np.transpose([fs, bs]))
    with open(oo.ddirname+'/bandpasses.txt', 'w') as f:
        f.write(stout)

def get_beams(oo):
    ls = np.arange(3*nside)
    stout = ''
    freqs = ut.get_freqs()
    for i_f, f in enumerate(freqs):
        b = np.ones(3*nside)
        fname = oo.ddirname+'/beam_%d.txt' % i_f
        stout += fname + '\n'
        np.savetxt(fname, np.transpose([ls, b]))
    with open(oo.ddirname+'/beams.txt', 'w') as f:
        f.write(stout)

def get_bpw_edges(oo):
    edges = np.arange(2, 304, 10, dtype=int)
    np.savetxt(oo.ddirname+'/bpw_edges.txt', edges,
               fmt='%d')

import yaml

ini_yml = get_ini(o)
with open(o.odirname+'/init.yaml', 'w') as f:
    yaml.dump(ini_yml, f, default_flow_style=False)
config_yml = get_config(o)
with open(o.odirname+'/config.yaml', 'w') as f:
    yaml.dump(config_yml, f, default_flow_style=False)

# Bandpasses
get_bandpasses(o)

# Beams
get_beams(o)

# Bpw edges
get_bpw_edges(o)

# Make masks
npix = hp.nside2npix(nside)
msks = np.ones([6, npix])
hp.write_map(o.ddirname+'/mask.fits', msks, overwrite=True)
