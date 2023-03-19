import numpy as np
import os
from tqdm import trange
import matplotlib.pyplot as plt
from WI_Solver_utils import InflatonModel, Background, Perturbations, Growth_factor
T_INIT = 6
T_END = -1
N = 100000
# DT is a negative quantity as expected since tau is decreasing over time
DT = (T_END - T_INIT) / N
TS = np.linspace(T_INIT, T_END, N)

Mpl = 1
g = 228.27  # SUSY relativistic degrees of freedom
a1 = np.pi**2/30*g

Neinflation= 60
Nruns=700


_,R2m_c3m0,R2std_c3m0=np.loadtxt('/home/gabriele/Desktop/PyCharm_env/WI_perturbations/WI_PowerSpectra_data/PowerSpectra_minimal_c3m0_n100000_Nr700.txt')
_,R2m_c3m2,R2std_c3m2=np.loadtxt('/home/gabriele/Desktop/PyCharm_env/WI_perturbations/WI_PowerSpectra_data/PowerSpectra_minimal_c3m0_n100000_Nr700.txt')

potential_type='minimal'
ICs_Q0_ph0=np.loadtxt('ICS_ph0-Q0s/ICS_ph0-Q0s_'+potential_type+'.txt')
Q0s=ICs_Q0_ph0[:,0]
ph0s=ICs_Q0_ph0[:,1]

#Example 1: Minimal Warm Inflation

M_sigma = 10**(6)/(2.42*10**(18))  # mass of waterfall field in units of Mpl
m_phi = 10**(-2)/(2.42*10**(18))  # mass of inflaton in units of Mpl
lv = 10**(-2)
Model = InflatonModel('minimal', [M_sigma, m_phi, lv], g, a1, Mpl)

Gf=Growth_factor(Model,Q0s,ph0s,R2m_c3m0,R2std_c3m0,Nruns,71,10**(6),TS,Neinflation,3,0)
popt,perr,Gf_sigc3m0,Gf_errc3m0=Gf.growth_factor_fit(method='complex',output_data=True)
popt2,perr2,_,_=Gf.growth_factor_fit(method='simple',output_data=True)
popt_jn=[2.22069485e+00, 8.71954145e-01, 1.88687870e+00, 5.19640757e+00,1.30335737e-01, 9.78162557e+00, 1.78877456e-09, 5.50983532e-01, 3.85067124e+00, 7.59917319e-07, 9.24061031e+00, 3.52588089e-14]

#Make Figure where I plot the data and the fit:

plt.figure()
plt.errorbar(Q0s,Gf_sigc3m0,yerr=Gf_errc3m0,c='k',label='Data',marker='.',ls='none',markersize=2)
plt.plot(Q0s,Gf.growth_factor_fit_func_positive_c_complex(Q0s,*popt),label='Complex Fit',c='tab:green',linewidth=0.7)
plt.plot(Q0s,Gf.growth_factor_fit_func_positive_c_simple(Q0s,*popt2),label='Simple Fit',c='tab:blue',linewidth=0.7)
plt.legend()
plt.xlabel(r'$Q$')
plt.ylabel(r'$G(Q)$')
plt.yscale('log')
plt.xscale('log')
#plt.ylim(10**(-4),10)
#plt.xlim(10**(-6),1)
plt.show()
print(popt)
print(perr)
