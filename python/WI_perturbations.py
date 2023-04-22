import numpy as np
import os
import time
from tqdm import trange
from WI_Solver_utils import InflatonModel, Background, Perturbations
import argparse

#from WI_Solver_utils_nompi import InflatonModel, Background, Perturbations

#Home directtory:
home_dir='WI_perturbations/'

'''
#Global parameters:

N = 100000
Nruns=1024
n_cores=128
Neinflation= 40

metric_perturbations=False
MPs_bool='wo' #without metric perturbations
negligible_epsH_etaH=True
epsH_etaH_bool='wo' #negligible epsH and etaH

'''

#Create parser to read command line arguments:

parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, default=100000, help='Number of time steps')
parser.add_argument('--Nruns', type=int, default=1024, help='Number of runs')
parser.add_argument('--n_cores', type=int, default=128, help='Number of cores')
parser.add_argument('--Neinflation', type=int, default=60, help='Number of e-folds of inflation')
# arguments for metric perturbations and negligible epsH and etaH as a string:
parser.add_argument('--metric_perturbations', type=str, default='False', help='are the metric perturbations relevant? True or False')
parser.add_argument('--negligible_epsH_etaH', type=str, default='True', help='are epsH and etaH negligible? True or False')

args = parser.parse_args()

N=args.N
Nruns=args.Nruns
n_cores=args.n_cores
Neinflation=args.Neinflation
metric_perturbations=args.metric_perturbations
negligible_epsH_etaH=args.negligible_epsH_etaH


MPs_bool='wo' #without metric perturbations
epsH_etaH_bool='wo' #negligible epsH and etaH

if metric_perturbations=='True':
    MPs_bool = 'w' #with metric perturbations
if negligible_epsH_etaH=='False':
    epsH_etaH_bool = 'w' #non-negligible epsH and etaH
r''' Time array where we take TS\equiv\tau= \ln(k/aH) '''

T_INIT = 6
T_END = -1
DT = (T_END - T_INIT) / N # DT is a negative quantity as expected since tau is decreasing over time
TS = np.linspace(T_INIT, T_END, N)

Mpl = 1
g = 228.27  # SUSY relativistic degrees of freedom
a1 = np.pi**2/30*g

if not os.path.exists(home_dir+'WI_PowerSpectra_data'):
    os.makedirs(home_dir+'WI_PowerSpectra_data')
potential_type='quartic'

ICs_Q0_ph0=np.loadtxt(home_dir+'ICS_ph0-Q0s/ICS_ph0-Q0s_'+potential_type+'_Ne'+str(int(Neinflation))+'.txt')


#Example 1: Quartic Potential

lv = 10**(-14)  # value of lambda

if potential_type=='quartic':
    Model = InflatonModel('monomial', [lv, 4], g, a1, Mpl)



Q0s=ICs_Q0_ph0[:,0]
ph0s=ICs_Q0_ph0[:,1]

cm_cases=[[0,0],[0,-1],[3,2],[3,0],[1,0],[-1,0],[-1,2]]

#cm_cases=[[0,0]]


l=0
while l in range(len(cm_cases)):
    cval,mval=cm_cases[l]
    numQs=len(Q0s)
    print('Case c='+str(int(cval))+' initiated')
    file_name=home_dir+'/WI_PowerSpectra_data/PowerSpectra_'+str(MPs_bool)+'MPs_'+str(epsH_etaH_bool)+'epsH-etaH_'+potential_type+'_c'+str(cval)+'m'+str(mval)+'_n'+str(int(N))+'_Nr'+str(int(Nruns))+'_Ne'+str(int(Neinflation))+'.txt'
    k=0
    while k in range(numQs):
        Bg = Background(Model, ph0s[k], Q0s[k])
        Bg_data=Bg.Bg_solver_tau(int(Neinflation+11), 10**(6), TS, Neinflation)
        Ps=Perturbations(Model,N,Nruns,TS,DT,cval,mval,Q0s[k],Bg_data)
        hatR2info= np.array([Ps.Pool_solver(range(n_cores),n_cores) for i in range(int(Nruns/n_cores))])
        #hatR2info is a n_cores x 2 array, where the first column is the hatR2 values and the second column is the standard deviation of the hatR2 values,
        # I want to compute the overall mean and standard deviation of the hatR2 values:
        hatR2=np.mean(hatR2info[:,0])   #mean of the hatR2 values
        hatR2std=np.sqrt(np.sum(hatR2info[:,1]**2))/len(hatR2info[:,1]) #standard deviation of the hatR2 values
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


