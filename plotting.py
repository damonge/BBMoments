import matplotlib.pyplot as plt
import numpy as np

dir0 = 'simulations_only_dust/sims_ns256_seed1000_lm383'
dir1 = 'simulations_only_dust/sims_ns256_seed1001_lm383'
dir2 = 'simulations_only_dust/sims_ns256_seed1002_lm383'
dir3 = 'simulations_only_dust/sims_ns256_seed1003_lm383'
dir4 = 'simulations_only_dust/sims_ns256_seed1004_lm383'
dir5 = 'simulations_only_dust/sims_ns256_seed1005_lm383'
dir6 = 'simulations_only_dust/sims_ns256_seed1006_lm383'
dir7 = 'simulations_only_dust/sims_ns256_seed1007_lm383'
dir8 = 'simulations_only_dust/sims_ns256_seed1008_lm383'
dir9 = 'simulations_only_dust/sims_ns256_seed1009_lm383'

ells = np.loadtxt(dir0+'/ells.txt')

def cl(dirname, typ):
    """ Import cl data from txt file """
    cl = np.zeros([51, 6, 6])
    for i_nu1 in range(6):  
        for i_nu2 in range(i_nu1, 6):
            if typ=='data_p0':
                cl[:,i_nu1,i_nu2] = np.loadtxt(dirname+'/cl_d_b0p0_%d_%d.txt'% (i_nu1, i_nu2))
            elif typ=='data_p2':
                cl[:,i_nu1,i_nu2] = np.loadtxt(dirname+'/cl_d_b0p2_%d_%d.txt'% (i_nu1, i_nu2))
            elif typ=='err_p0':
                cl[:,i_nu1,i_nu2] = np.loadtxt(dirname+'/err_b0p0_%d_%d.txt'% (i_nu1, i_nu2))
            elif typ=='err_p2':
                cl[:,i_nu1,i_nu2] = np.loadtxt(dirname+'/err_b0p2_%d_%d.txt'% (i_nu1, i_nu2))
            elif typ=='theory_p0':
                cl[:,i_nu1,i_nu2] = np.loadtxt(dirname+'/cl_t_b0p0_%d_%d.txt'% (i_nu1, i_nu2))
            elif typ=='theory_p2':
                cl[:,i_nu1,i_nu2] = np.loadtxt(dirname+'/cl_t_b0p2_%d_%d.txt'% (i_nu1, i_nu2))
    return cl

def cl_mean(cl0, cl1, cl2, cl3, cl4, cl5, cl6, cl7, cl8, cl9):
    """ Function to calculate mean of the 10 realizations """
    cl_mean_01 = np.add(cl0, cl1)
    cl_mean_23 = np.add(cl2, cl3)
    cl_mean_45 = np.add(cl4, cl5)
    cl_mean_67 = np.add(cl6, cl7)
    cl_mean_89 = np.add(cl8, cl9)
    cl_mean_0123 = np.add(cl_mean_01, cl_mean_23)
    cl_mean_4567 = np.add(cl_mean_45, cl_mean_67)
    cl_mean_01234567 = np.add(cl_mean_0123, cl_mean_4567)
    cl_mean_all = np.add(cl_mean_89, cl_mean_01234567)
    return cl_mean_all / 10

# Calculate the mean cls for
#
# Data
cl_d_b0p0_mean = cl_mean(cl(dir0, 'data_p0'), cl(dir1, 'data_p0'), cl(dir2, 'data_p0'), cl(dir3, 'data_p0'), cl(dir4, 'data_p0'), cl(dir5, 'data_p0'), cl(dir6, 'data_p0'), cl(dir7, 'data_p0'), cl(dir8, 'data_p0'), cl(dir9, 'data_p0')) 

cl_d_b0p2_mean = cl_mean(cl(dir0, 'data_p2'), cl(dir1, 'data_p2'), cl(dir2, 'data_p2'), cl(dir3, 'data_p2'), cl(dir4, 'data_p2'), cl(dir5, 'data_p2'), cl(dir6, 'data_p2'), cl(dir7, 'data_p2'), cl(dir8, 'data_p2'), cl(dir9, 'data_p2'))
#
# Theory
cl_t_b0p0_mean = cl_mean(cl(dir0, 'theory_p0'), cl(dir1, 'theory_p0'), cl(dir2, 'theory_p0'), cl(dir3, 'theory_p0'), cl(dir4, 'theory_p0'), cl(dir5, 'theory_p0'), cl(dir6, 'theory_p0'), cl(dir7, 'theory_p0'), cl(dir8, 'theory_p0'), cl(dir9, 'theory_p0'))

cl_t_b0p2_mean = cl_mean(cl(dir0, 'theory_p2'), cl(dir1, 'theory_p2'), cl(dir2, 'theory_p2'), cl(dir3, 'theory_p2'), cl(dir4, 'theory_p2'), cl(dir5, 'theory_p2'), cl(dir6, 'theory_p2'), cl(dir7, 'theory_p2'), cl(dir8, 'theory_p2'), cl(dir9, 'theory_p2'))
#
# Errors
err_b0p0_mean = cl_mean(cl(dir0, 'err_p0'), cl(dir1, 'err_p0'), cl(dir2, 'err_p0'), cl(dir3, 'err_p0'), cl(dir4, 'err_p0'), cl(dir5, 'err_p0'), cl(dir6, 'err_p0'), cl(dir7, 'err_p0'), cl(dir8, 'err_p0'), cl(dir9, 'err_p0'))

err_b0p2_mean = cl_mean(cl(dir0, 'err_p2'), cl(dir1, 'err_p2'), cl(dir2, 'err_p2'), cl(dir3, 'err_p2'), cl(dir4, 'err_p2'), cl(dir5, 'err_p2'), cl(dir6, 'err_p2'), cl(dir7, 'err_p2'), cl(dir8, 'err_p2'), cl(dir9, 'err_p2'))

# Plot stuff
#
# All plots together
fig, ax = plt.subplots(6, 6, figsize=(38,40), sharex='col', sharey='row')
#fig, ax = plt.subplots(6, 6, sharex='col', sharey='row')
fig.subplots_adjust(hspace=0.9, wspace=0.7)
for i_1 in range(6):  
    for i_2 in range(i_1,6):
        ax[i_1,i_2].set_title(str('%d x %d' % (i_1, i_2)), fontsize=9)
        ax[i_1,i_2].errorbar(ells, cl_d_b0p0_mean[:,i_1,i_2], yerr=err_b0p0_mean[:,i_1,i_2],
                     fmt='k.', label='No beta variations')
        ax[i_1,i_2].plot(ells, cl_t_b0p0_mean[:,i_1,i_2], 'k--')
        ax[i_1,i_2].errorbar(ells, cl_d_b0p2_mean[:,i_1,i_2], yerr=err_b0p2_mean[:,i_1,i_2],
                     fmt='r.', label='0.2 beta variations')
        ax[i_1,i_2].plot(ells, cl_t_b0p2_mean[:,i_1,i_2], 'r--')
        ax[i_1,i_2].set_yscale('log')
        ax[i_1,i_2].set_xscale('log') 
        ax[i_1,i_2].set_xlabel(str(r'$\ell$'), fontsize=8)
        ax[i_1,i_2].set_ylabel(str(r'$D_\ell$'), fontsize=8)
plt.savefig(f'plots/mean_all_loglog.pdf')
#
# Single plots
for i_1 in range(6):  
    for i_2 in range(i_1,6):
        plt.figure()
        plt.title('%d x %d' % (i_1, i_2), fontsize=14)
        plt.errorbar(ells, cl_d_b0p0_mean[:,i_1,i_2], yerr=err_b0p0_mean[:,i_1,i_2],
                     fmt='k.', label='No beta variations')
        plt.plot(ells, cl_t_b0p0_mean[:,i_1,i_2], 'k--')
        plt.errorbar(ells, cl_d_b0p2_mean[:,i_1,i_2], yerr=err_b0p2_mean[:,i_1,i_2],
                     fmt='r.', label='0.2 beta variations')
        plt.plot(ells, cl_t_b0p2_mean[:,i_1,i_2], 'r--')
        plt.xlabel(r'$\ell$', fontsize=16)
        plt.ylabel(r'$D_\ell$', fontsize=16)
        plt.loglog()
        plt.legend()
        plt.savefig(f'plots/mean_only_dust_{i_1}_{i_2}.png')

