import pymaster as nmt
import healpy as hp
import numpy as np
import os
import scipy.constants as constants
from scipy import optimize
import matplotlib.pyplot as plt
from optparse import OptionParser
from optparse import OptionParser

# Masked amplitude maps to power spectra
def map2dl(mask, maps, i, j, k, l):
    # Read amplitude maps
    q_nu1, u_nu1 = hp.ud_grade(hp.read_map(maps, verbose=False, field=[i,j]), nside_out=512) 
    q_nu2, u_nu2 = hp.ud_grade(hp.read_map(maps, verbose=False, field=[k,l]), nside_out=512) 
    nside = hp.npix2nside(len(q_nu1))
    # Binary mask
    mask_binary = np.ones_like(mask)
    mask_binary[mask<=0] = 0
    # Mask fields
    q_nu1 *= mask_binary
    u_nu1 *= mask_binary
    q_nu2 *= mask_binary
    u_nu2 *= mask_binary
    # Binning
    # nlb = beam_width = 10 ells
    bn = nmt.NmtBin(nside, nlb=10, is_Dell=True)
    # Field
    field1 = nmt.NmtField(mask, [q_nu1, u_nu1])
    field2 = nmt.NmtField(mask, [q_nu2, u_nu2])
    # M-C-M (Mode Coupling Matrix)
    wsp = nmt.NmtWorkspace()
    if os.path.isfile("mcm.fits"):
        wsp.read_from("mcm.fits")
    else:
        wsp.compute_coupling_matrix(field1, field2, bn)
        wsp.write_to("mcm.fits")
    # C_ell
    dl_ee, dl_eb, dl_be, dl_bb = wsp.decouple_cell(nmt.compute_coupled_cell(field1, field2))
    ell_eff = bn.get_effective_ells()
    return ell_eff, dl_ee, dl_bb

# Read mask
mask = hp.read_map("./data/masks_SAT.fits", verbose=False)

# Maps of synchrotron, dust and sync+dust
maps_gaus_s='./FITS_sim_ns256_seed1000_stdd0_stds0_gdm3.0_gsm3.0_msk_E_B_nu0d353_nu0s23_sync_Ad28/maps_sky_signal.fits'
maps_pysm_s='./FITS_sim_ns256_seed1000_stdd0_stds0_gdm3.0_gsm3.0_msk_E_B_pysmBetas_realAmps_nu0d353_nu0s23_sync_Ad5/maps_sky_signal.fits'
maps_gaus_d='./FITS_sim_ns256_seed1000_stdd0_stds0_gdm3.0_gsm3.0_msk_E_B_nu0d353_nu0s23_dust_Ad28/maps_sky_signal.fits'
maps_pysm_d='./FITS_sim_ns256_seed1000_stdd0_stds0_gdm3.0_gsm3.0_msk_E_B_pysmBetas_realAmps_nu0d353_nu0s23_dust_Ad5/maps_sky_signal.fits'
maps_gaus_ds='./FITS_sim_ns256_seed1000_stdd0_stds0_gdm3.0_gsm3.0_msk_E_B_nu0d353_nu0s23_dust_sync_Ad28/maps_sky_signal.fits'
maps_pysm_ds='./FITS_sim_ns256_seed1000_stdd0_stds0_gdm3.0_gsm3.0_msk_E_B_pysmBetas_realAmps_nu0d353_nu0s23_dust_sync_Ad5/maps_sky_signal.fits'

if not os.path.isfile("Outputs_compare_cls_Pysm_vs_Gaus/dls_gaus_s.txt") and os.path.isfile("Outputs_compare_cls_Pysm_vs_Gaus/dls_gaus_d.txt") and os.path.isfile("Outputs_compare_cls_Pysm_vs_Gaus/dls_gaus_ds.txt") and os.path.isfile("Outputs_compare_cls_Pysm_vs_Gaus/ls.txt"):
    open("Outputs_compare_cls_Pysm_vs_Gaus/ls.txt", 'w')
    open('Outputs_compare_cls_Pysm_vs_Gaus/dls_gaus_s.txt', 'w')                                                                                             
    open('Outputs_compare_cls_Pysm_vs_Gaus/dls_gaus_d.txt', 'w')                                                                                             
    open('Outputs_compare_cls_Pysm_vs_Gaus/dls_gaus_ds.txt', 'w')                                                                                       
    open('Outputs_compare_cls_Pysm_vs_Gaus/dls_pysm_s.txt', 'w')                                                                                             
    open('Outputs_compare_cls_Pysm_vs_Gaus/dls_pysm_d.txt', 'w')                                                                                             
    open('Outputs_compare_cls_Pysm_vs_Gaus/dls_pysm_ds.txt', 'w')
    dls_gaus_s = []
    dls_pysm_s = []
    dls_gaus_d = []
    dls_pysm_d = []
    dls_gaus_ds = []
    dls_pysm_ds = []
    ls = []
    for i in np.arange(0, 12, 2):
        j=i+1
        for k in np.arange(i, 12, 2):
            l=k+1
            ''' Write the dls arrrays to disk
            order: 1x1, 1x2, 1x3, 1x4, 1x5, 1x6,
                        2x2, 2x3, 2x4, 2x5, 2x6,
                             3x3, 3x4, 3x5, 3x6,
                                  4x4, 4x5, 4x6,
                                       5x5, 5x6,
                                            6x6.'''        
            print('Selecting map fields')
            print('i j k l')
            print(i, j, k, l)
            print('Computing Dl at cross frequencies')        
            print('nu1 nu2')
            print((i+2)//2, (k+2)//2)
            print('-------------------')
        
            ell_eff, dl_ee_pysm_s, dl_bb_pysm_s = map2dl(mask, maps_pysm_s, i,j,k,l)
            _, dl_ee_gaus_s, dl_bb_gaus_s = map2dl(mask, maps_gaus_s, i,j,k,l)
        
            _, dl_ee_pysm_d, dl_bb_pysm_d = map2dl(mask, maps_pysm_d, i,j,k,l)
            _, dl_ee_gaus_d, dl_bb_gaus_d = map2dl(mask, maps_gaus_d, i,j,k,l)
        
            _, dl_ee_pysm_ds, dl_bb_pysm_ds = map2dl(mask, maps_pysm_ds,i,j,k,l)
            _, dl_ee_gaus_ds, dl_bb_gaus_ds = map2dl(mask, maps_gaus_ds, i,j,k,l)
    
            ells=ell_eff
            dl2cl = np.ones(len(ells))
            dl2cl[1:] = 2*np.pi/(ells[1:]*(ells[1:]+1.))
            cl2dl = (ells*(ells+1.))/(2*np.pi)
            #cl_bb_gaus *= dl2cl
            #cl_bb_pysm *= dl2cl
        
            dls_gaus_s.append(dl_bb_gaus_s)
            dls_pysm_s.append(dl_bb_pysm_s)
            dls_gaus_d.append(dl_bb_gaus_d)
            dls_pysm_d.append(dl_bb_pysm_d)
            dls_gaus_ds.append(dl_bb_gaus_ds)
            dls_pysm_ds.append(dl_bb_pysm_ds)
            ls.append(ell_eff)
    
    ls = np.asarray(ls)
    dls_gaus_s = np.asarray(dls_gaus_s)
    dls_pysm_s = np.asarray(dls_pysm_s)
    dls_gaus_d = np.asarray(dls_gaus_d)
    dls_pysm_d = np.asarray(dls_pysm_d)
    dls_gaus_ds = np.asarray(dls_gaus_ds)
    dls_pysm_ds = np.asarray(dls_pysm_ds)
    
    np.savetxt("Outputs_compare_cls_Pysm_vs_Gaus/ls.txt", ls)
    np.savetxt("Outputs_compare_cls_Pysm_vs_Gaus/dls_gaus_s.txt", dls_gaus_s)
    np.savetxt("Outputs_compare_cls_Pysm_vs_Gaus/dls_pysm_s.txt", dls_pysm_s)
    np.savetxt("Outputs_compare_cls_Pysm_vs_Gaus/dls_gaus_d.txt", dls_gaus_d)
    np.savetxt("Outputs_compare_cls_Pysm_vs_Gaus/dls_pysm_d.txt", dls_pysm_d)
    np.savetxt("Outputs_compare_cls_Pysm_vs_Gaus/dls_gaus_ds.txt", dls_gaus_ds)
    np.savetxt("Outputs_compare_cls_Pysm_vs_Gaus/dls_pysm_ds.txt", dls_pysm_ds)
else:
    ls = np.loadtxt("Outputs_compare_cls_Pysm_vs_Gaus/ls.txt")
    dls_gaus_s = np.loadtxt("Outputs_compare_cls_Pysm_vs_Gaus/dls_gaus_s.txt")
    dls_pysm_s = np.loadtxt("Outputs_compare_cls_Pysm_vs_Gaus/dls_pysm_s.txt")
    dls_gaus_d = np.loadtxt("Outputs_compare_cls_Pysm_vs_Gaus/dls_gaus_d.txt")
    dls_pysm_d = np.loadtxt("Outputs_compare_cls_Pysm_vs_Gaus/dls_pysm_d.txt")
    dls_gaus_ds = np.loadtxt("Outputs_compare_cls_Pysm_vs_Gaus/dls_gaus_ds.txt")
    dls_pysm_ds = np.loadtxt("Outputs_compare_cls_Pysm_vs_Gaus/dls_pysm_ds.txt")

# Plot results
label_list = []
for i in np.arange(0, 12, 2):
    for k in np.arange(i, 12, 2):
        label_list.append('%s_x_%s'%((i+2)//2,(k+2)//2))

for i in np.arange(21):
    labels = label_list[i]    
    plt.figure()
    plt.plot(ls[i][3:30], dls_gaus_s[i][3:30], 'r-', label='gaussian sim')
    plt.plot(ls[i][3:30], dls_pysm_s[i][3:30], 'b-', label='pysm sim')
    plt.title('Synchrotron'+label_list[i]) 
    plt.loglog()
    plt.xlabel('$\\ell$', fontsize=16)
    plt.ylabel('$D_\\ell$', fontsize=16)
    plt.legend(loc='upper right', ncol=2, labelspacing=0.1)
    plt.savefig('Outputs_compare_cls_Pysm_vs_Gaus/dlbb_SIMsync_COMPARA_'+label_list[i]) 

    plt.figure()
    plt.plot(ls[i][3:30], dls_gaus_d[i][3:30], 'r-', label='gaussian sim')
    plt.plot(ls[i][3:30], dls_pysm_d[i][3:30], 'b-', label='pysm sim')
    plt.title('Dust'+label_list[i]) 
    plt.loglog()
    plt.xlabel('$\\ell$', fontsize=16)
    plt.ylabel('$D_\\ell$', fontsize=16)
    plt.legend(loc='upper right', ncol=2, labelspacing=0.1)
    plt.savefig('Outputs_compare_cls_Pysm_vs_Gaus/dlbb_SIMdust_COMPARA_'+label_list[i]) 

    plt.figure()
    plt.plot(ls[i][3:30], dls_gaus_ds[i][3:30], 'r-', label='gaussian sim')
    plt.plot(ls[i][3:30], dls_pysm_ds[i][3:30], 'b-', label='pysm sim')
    plt.title('Dust + Synchrotron'+label_list[i]) 
    plt.loglog()
    plt.xlabel('$\\ell$', fontsize=16)
    plt.ylabel('$D_\\ell$', fontsize=16)
    plt.legend(loc='upper right', ncol=2, labelspacing=0.1)
    plt.savefig('Outputs_compare_cls_Pysm_vs_Gaus/dlbb_SIMdustsync_COMPARA_'+label_list[i]) 




