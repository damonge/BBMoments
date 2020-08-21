import os
import numpy as np
import matplotlib.pyplot as plt
import utils as ut
from scipy.interpolate import interp1d
from hankel import HankelTransform


A_dust = 28.
alpha_dust = -0.5
A_beta = 5E-6
gamma = -2.5
beta_dust = 1.6
temp_dust = 19.6

nu_0 = 353.
nu_1 = 39.
nu_2 = 39.

sed_1 = ut.comp_sed(nu_1, nu_0, beta_dust, temp_dust, 'dust')
sed_2 = ut.comp_sed(nu_2, nu_0, beta_dust, temp_dust, 'dust')
lognu_1 = np.log(nu_1/nu_0)
lognu_2 = np.log(nu_2/nu_0)


def cl_amp_f(ell):
    return A_dust * ((ell+5.)/80)**(-2+alpha_dust)


def cl_beta_f(ell):
    return A_beta * ((ell+5.)/80)**gamma


# First we compute the moments expansion prediction
nside = 256
ls = np.arange(3*nside)
cl_amp = cl_amp_f(ls)
cl_beta = cl_beta_f(ls)
w3j = ut.get_w3j()
cl_1x1 = ut.w3j_sandwich(ls, cl_amp, cl_beta, w3j)
sig2_beta = np.sum((2*ls+1)*cl_beta)/(4*np.pi)
cl_0x2 = sig2_beta * cl_amp
sed_prod = sed_1 * sed_2
cl_0x0 = cl_amp * sed_prod
cl_1x1 = cl_1x1 * sed_prod * lognu_1 * lognu_2
cl_0x2 = cl_0x2 * sed_prod * 0.5 * (lognu_1**2 + lognu_2**2)
d_mom = {'ls': ls,
         'cl_0x0': cl_0x0,
         'cl_1x1': cl_1x1,
         'cl_0x2': cl_0x2}

# Now let's compute the result of 100 sims
if not os.path.isfile("cl_sim.npz"):
    import healpy as hp
    nside = 256
    nsim = 100
    ls = np.arange(3*nside)
    cl_amp = cl_amp_f(ls)
    cl_beta = cl_beta_f(ls)
    cl_const = np.zeros(3*nside)
    cl_var = np.zeros(3*nside)
    for i in range(nsim):
        print(i)
        map_amp = hp.synfast(cl_amp, nside, verbose=False)
        map_beta = beta_dust + hp.synfast(cl_beta, nside, verbose=False)
        s1 = ut.comp_sed(nu_1, nu_0, map_beta, temp_dust, 'dust')
        s2 = ut.comp_sed(nu_2, nu_0, map_beta, temp_dust, 'dust')
        m1 = map_amp*s1
        m2 = map_amp*s2
        cl_const += hp.anafast(map_amp, iter=0) * sed_1 * sed_2
        cl_var += hp.anafast(m1, m2, iter=0)
    cl_const /= nsim
    cl_var /= nsim
    np.savez("cl_sim.npz", ls=ls, cl_const=cl_const, cl_var=cl_var)
d_sim = np.load("cl_sim.npz")


# Now let's compute xi_Amp and xi_beta
ht = HankelTransform(
    nu= 0,     # The order of the bessel function
    N = 100,   # Number of steps in the integration
    h = 0.0003   # Proxy for "size" of steps in integration
)
th = np.geomspace(1E-5, 1E2, 1000)
# xi_Amplitude
xi_amp = ht.transform(cl_amp_f, th, ret_err=False) / (2 * np.pi)
xi_amp *= sed_1 * sed_2
# xi_beta
xi_beta = ht.transform(cl_beta_f, th, ret_err=False) / (2 * np.pi)

# We'll probably need to evaluate xi(theta) for theta beyond the
# range of values we've calculated. For this, let's create a function
# that interpolates within the values we have actually computed and
# then extrapolates as a power law outside of that range.
# 1- Interpolation
xi_amp_i = interp1d(np.log(th), np.log(xi_amp), fill_value=0, bounds_error=False)
xi_beta_i = interp1d(np.log(th), np.log(xi_beta), fill_value=0, bounds_error=False)
# 2- Logarithmic slopes on small and large theta for extrapolation
xi_amp_m0 = np.log(xi_amp[1]/xi_amp[0])/np.log(th[1]/th[0])
xi_amp_mf = np.log(xi_amp[-1]/xi_amp[-2])/np.log(th[-1]/th[-2])
xi_beta_m0 = np.log(xi_beta[1]/xi_beta[0])/np.log(th[1]/th[0])
xi_beta_mf = np.log(xi_beta[-1]/xi_beta[-2])/np.log(th[-1]/th[-2])

# Xi inter/extrapolation function
def xi_f(t, is_amp=True):
    if is_amp:
        xi = xi_amp
        xi_m0 = xi_amp_m0
        xi_mf = xi_amp_mf
        xi_i = xi_amp_i
    else:
        xi = xi_beta
        xi_m0 = xi_beta_m0
        xi_mf = xi_beta_mf
        xi_i = xi_beta_i

    def xi_f0(theta):
        # Extrapolation on high-theta
        return xi[0] * np.exp(xi_m0 * np.log(theta/th[0]))

    def xi_ff(theta):
        # Extrapolation on low-theta
        return xi[-1] * np.exp(xi_mf * np.log(theta/th[-1]))

    def xi_fm(theta):
        # Interpolation on intermediate theta
        return np.exp(xi_i(np.log(theta)))

    return np.piecewise(t,
                        [t < th[0],
                         (t >= th[0]) * (t <= th[-1]),
                         t > th[-1]],
                        [xi_f0, xi_fm, xi_ff])

# We also need sigma_beta
ls = np.arange(100000)
sigma_beta = np.sqrt(np.sum((2*ls+1)*cl_beta_f(ls)/(4*np.pi)))

# This function returns the correlation function
# accounting for beta fluctuations.
def xi_combined(t):
    # xi_amplitudes
    xi_A = xi_f(t, True)
    # xi_beta
    xi_b = xi_f(t, False)
    # sigma_beta^2
    s2_b = sigma_beta**2
    return xi_A * np.exp(s2_b*0.5*(lognu_1**2+lognu_2**2) +
                         lognu_1*lognu_2*xi_b)


# Now let's transform back to C_ell
l = np.linspace(2, 1000, 1000)
# C_ell without beta variations
cl_const = sed_1*sed_2*cl_amp_f(l)
ht = HankelTransform(
    nu= 0,     # The order of the bessel function
    N = 1000,   # Number of steps in the integration
    h = 0.003   # Proxy for "size" of steps in integration
)
cl_var = ht.transform(xi_combined, l, ret_err=False)*2*np.pi
# And let's just test that we can recover the input
# C_ell from xi.
cl_back = ht.transform(xi_f, l, ret_err=False)*2*np.pi


# Plot the results!
plt.figure()
def plot_cl(ls, cls, lstyle, l0=30, lf=300, label=''):
    msk = (ls <= lf) & (ls >= l0)
    lfac = ls * (ls+1) / (2 * np.pi)
    plt.plot(ls[msk], (lfac*cls)[msk], lstyle, label=label)
plot_cl(d_sim['ls'], d_sim['cl_const'], 'k--', label=r'Sim., $\delta\beta=0$')
plot_cl(d_sim['ls'], d_sim['cl_var'], 'r--', label=r'Sim., $\delta\beta\neq 0$')
plot_cl(d_mom['ls'], d_mom['cl_0x0']+d_mom['cl_1x1']+d_mom['cl_0x2'], 'b--',
        label=r'$C_\ell^{0\times 0}+C_\ell^{1\times 1}+C_\ell^{0\times 2}$')
plot_cl(l, cl_const, 'k-', label=r'$C_\ell^{0\times 0}$')
plot_cl(l, cl_var, 'r-', label=r'$C_\ell^{\rm NP}$')
plot_cl(l, cl_back, 'y:')
plt.xlabel(r'$\ell$', fontsize=15)
plt.ylabel(r'$D_\ell^{39\times 39}\,\,[\mu K_{\rm CMB}^2]$', fontsize=15)
plt.legend(loc='upper right')
plt.savefig('xi_cl.png', bbox_inches='tight')
plt.show()
