import numpy as np
import os
from tqdm import trange
from WI_Solver_utils import InflatonModel, Background, Perturbations
T_INIT = 6
T_END = -1
N = 100000
# DT is a negative quantity as expected since tau is decreasing over time
DT = (T_END - T_INIT) / N
TS = np.linspace(T_INIT, T_END, N)

Mpl = 1
g = 228.27  # SUSY relativistic degrees of freedom
a1 = np.pi**2/30*g

############################################################################################################
###                             EXAMPLE 1: MINIMAL WARM INFLATION                                        ###
############################################################################################################

# initial conditions
Q0s=[9.30000000e+03, 8.00000000e+03, 7.94328235e+03, 6.47308204e+03,5.27499706e+03, 5.00000000e+03, 4.29866235e+03, 3.98107000e+03, 3.50303474e+03, 2.85466766e+03, 2.32630507e+03, 1.89573565e+03, 1.58489000e+03, 1.54485915e+03, 1.25892541e+03, 1.00000000e+03, 6.60693448e+02, 6.30957000e+02, 4.36515832e+02, 2.88403150e+02, 2.51189000e+02, 1.90546072e+02, 1.25892541e+02, 1.00000000e+02, 8.60000000e+01, 7.60000000e+01, 6.60000000e+01, 5.60000000e+01, 4.60000000e+01, 3.60000000e+01, 2.60000000e+01, 1.60000000e+01,6.00000000e+00, 3.16227766e+00, 1.00000000e+00, 4.86967525e-01, 2.37137000e-01, 7.49894209e-02, 5.62341000e-02, 1.33352000e-02, 1.15478198e-02, 3.16228000e-03, 1.77827941e-03, 7.49894000e-04, 2.73841963e-04, 1.77828000e-04, 4.21697000e-05, 4.21696503e-05, 1.00000000e-05, 6.49381632e-06, 1.00000000e-06, 1.00000000e-07]
ph0s=[2.43000000e-01, 2.62000000e-01, 2.63285600e-01, 2.96610140e-01, 3.23766733e-01, 3.30000000e-01, 3.57532319e-01, 3.70000000e-01, 4.11894768e-01, 4.68717330e-01, 5.15022760e-01, 5.52757645e-01, 5.90000000e-01, 5.94687403e-01, 6.99597085e-01, 7.94597356e-01, 9.19089635e-01, 9.30000000e-01, 1.20647993e+00, 1.41708443e+00, 1.47000000e+00, 1.81495180e+00, 2.18271689e+00, 2.33000000e+00, 2.51000000e+00, 2.67000000e+00, 2.86000000e+00, 3.10000000e+00, 3.42000000e+00, 3.85000000e+00, 4.50000000e+00, 5.70000000e+00, 8.84000000e+00, 1.32441451e+01, 1.66000000e+01, 1.96262919e+01, 2.11000000e+01, 2.26237505e+01, 2.28000000e+01, 2.33000000e+01, 2.33175700e+01, 2.34000000e+01, 2.34573706e+01, 2.35000000e+01, 2.35000000e+01, 2.35000000e+01, 2.35000000e+01, 2.35000000e+01, 2.35000000e+01, 2.35000000e+01, 2.35000000e+01, 2.35000000e+01]
#Create a directory where to save the txt files with the initial conditions for the inflaton field, if the directory does not exist yet:
if not os.path.exists('ICS_ph0-Q0s'):
    os.makedirs('ICS_ph0-Q0s')
new_ICs=[]
'''
M_sigma = 10**(6)/(2.42*10**(18))  # mass of waterfall field in units of Mpl
m_phi = 10**(-2)/(2.42*10**(18))  # mass of inflaton in units of Mpl
lv = 10**(-2)
Model = InflatonModel('minimal', [M_sigma, m_phi, lv], g, a1, Mpl)

for i in trange(len(Q0s)):
    Bg = Background(Model, ph0s[i], Q0s[i])
    ph0_new = Bg.Bg_solver_test(71, 10**(6), 60, verbose=False, learning_rate=0.001)
    new_ICs.append([Q0s[i],ph0_new])
#Save the new initial conditions in a txt file:
np.savetxt('ICS_ph0-Q0s/ICS_ph0-Q0s_minimal.txt', new_ICs)
quit()

############################################################################################################
###                             EXAMPLE 2: MONOMIAL QUARTIC POTENTIAL                                    ###
############################################################################################################

lv = 10**(-14)  # value of lambda


Model = InflatonModel('monomial', [lv*np.math.factorial(4), 4], g, a1, Mpl)
i=0
while i in trange(len(Q0s)):
    Bg = Background(Model, ph0s[i], Q0s[i])
    ph0_new = Bg.Bg_solver_test(71, 10**(6), 60, verbose=False, learning_rate=0.03)
    new_ICs.append([Q0s[i],ph0_new])
#Save the new initial conditions in a txt file:
np.savetxt('ICS_ph0-Q0s/ICS_ph0-Q0s_quartic.txt', new_ICs)
quit()

############################################################################################################
###                             EXAMPLE 3: BETA EXPONENTIAL POTENTIAL                                    ###
############################################################################################################
beta=0.3
lv=0.05
V0=10**(-23)
Model = InflatonModel('beta_exponential', [V0,beta, lv], g, a1, Mpl)
i=0
while i in range(len(Q0s)):
    Bg = Background(Model, 10**(-10), 100)
    ph0_new = Bg.Bg_solver_test(71, 10**(6), 60, verbose=True, learning_rate=0.00001)
    #new_ICs.append([Q0s[i],ph0_new])
    break
'''
############################################################################################################
###                                      EXAMPLE 4: RUNAWAY POTENTIAL                                    ###
############################################################################################################
alpha=6
n=2
V0=1
Model = InflatonModel('runaway', [V0,alpha,n], g, a1, Mpl)
i=0
while i in range(len(Q0s)):
    Bg = Background(Model, .5, 500)
    ph0_new = Bg.Bg_solver_test(100, 10**(6), 90, verbose=True, learning_rate=1)
    #new_ICs.append([Q0s[i],ph0_new])
    break
