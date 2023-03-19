import numpy as np
import os
from tqdm import trange
from WI_Solver_utils import InflatonModel, Background, Perturbations


#Global parameters:

N = 100000
Nruns=700
Neinflation= 60

metric_perturbations=False
MPs_bool='wo' #without metric perturbations
negligible_epsH_etaH=True
epsH_etaH_bool='wo' #negligible epsH and etaH

if metric_perturbations:
    MPs_bool = 'w' #with metric perturbations
if not negligible_epsH_etaH:
    epsH_etaH_bool = 'w' #non-negligible epsH and etaH

r''' Time array where we take TS\equiv\tau= \ln(k/aH) '''

T_INIT = 6
T_END = -1
DT = (T_END - T_INIT) / N # DT is a negative quantity as expected since tau is decreasing over time
TS = np.linspace(T_INIT, T_END, N)

Mpl = 1
g = 228.27  # SUSY relativistic degrees of freedom
a1 = np.pi**2/30*g

if not os.path.exists('WI_PowerSpectra_data'):
    os.makedirs('WI_PowerSpectra_data')
potential_type='minimal'

ICs_Q0_ph0=np.loadtxt('ICS_ph0-Q0s/ICS_ph0-Q0s_'+potential_type+'.txt')


#Example 1: Minimal Warm Inflation

M_sigma = 10**(6)/(2.42*10**(18))  # mass of waterfall field in units of Mpl
m_phi = 10**(-2)/(2.42*10**(18))  # mass of inflaton in units of Mpl
lv = 10**(-2)
Model = InflatonModel(potential_type, [M_sigma, m_phi, lv], g, a1, Mpl)

'''

#Example 2: Quartic Potential

lv = 10**(-14)  # value of lambda

if potential_type=='quartic':
    Model = InflatonModel('monomial', [lv, 4], g, a1, Mpl)
'''


Q0s=ICs_Q0_ph0[:,0]
ph0s=ICs_Q0_ph0[:,1]


cm_cases=[[0,0],[0,-1],[3,2],[3,0],[1,0],[-1,0],[-1,2]]


l=0
while l in range(len(cm_cases)):
    cval,mval=cm_cases[l]
    numQs=len(Q0s)
    print('Case c='+str(int(cval))+' initiated')
    file_name='WI_PowerSpectra_data/PowerSpectra_'+str(MPs_bool)+'MPs_'+str(epsH_etaH_bool)+'epsH-etaH_'+potential_type+'_c'+str(cval)+'m'+str(mval)+'_n'+str(int(N))+'_Nr'+str(int(Nruns))+'_Ne'+str(int(Neinflation))+'.txt'
    k=0
    while k in range(numQs):
        Bg = Background(Model, ph0s[k], Q0s[k])
        Bg_data=Bg.Bg_solver_tau(71, 10**(6), TS, 60)
        Ps=Perturbations(Model,N,Nruns,TS,DT,cval,mval,Q0s[k],Bg_data)
        hatR2,hatR2std = Ps.solve(0)
        data1 = np.array([Q0s[k], hatR2, hatR2std])
        if not os.path.exists(file_name):
        # File doesn't exist, so create it with the new column
            datas = np.expand_dims(data1, axis=1)
        else:
        # File exists, so load the data and check if the new column already exists
            datas = np.loadtxt(file_name)
            if datas.ndim == 1:
                datas = datas.reshape(-1, 1)
        if not np.array_equal(datas[:, 0], data1):
            if not np.any(datas[0, :] == data1[0]):
            # New column doesn't exist, so add it as the first column
            # Only add if first element of new column not in first row
                location = np.searchsorted(-datas[0, :], -data1[0],'left')
                datas = np.insert(datas, location, data1, axis=1)
            else:
            # Substitute the new column for existing column
                idx = np.where(datas[0, :] == data1[0])[0][0]
                datas[:, idx] = data1
        # Save the data back to the file
        np.savetxt(file_name, datas)
        print('Q_0='+str(Q0s[k])+' done!')
        k+=1
    l+=1


