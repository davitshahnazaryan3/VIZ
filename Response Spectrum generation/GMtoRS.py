"""
Software to obtain response spectra from a record.

Written by Davit Shahnazaryan.

"""

from __future__ import division
import pandas as pd
import cmath
import numpy as np
import matplotlib.pyplot as plt
import timeit
from pathlib import Path
import pickle

directory = Path.cwd()

plt.rc('font', family='Times New Roman') 
plt.rc('font', serif='Times New Roman')

# Ignore these pesky warnings
import warnings
warnings.filterwarnings('ignore')
   
# Documenting duration of run
start_time = timeit.default_timer()

def _responseSpectrum(acc, dt, period, damping):
    if period == 0.0:
        period = 1e-20
    PGA = max(acc)
    pow = 1
    while (2 ** pow  < len(acc)):
        pow = pow + 1
    nPts = 2 ** pow
    fas = np.fft.fft(acc, nPts)
    dFreq = 1/(dt*(nPts - 1))
    freq = dFreq * np.array(range(nPts))
    if nPts%2 != 0:
        symIdx = int(np.ceil(nPts/2))
    else:
        symIdx = int(1 + nPts/2)
    natFreq = 1/period
    H = np.ones(len(fas), 'complex')
    H[np.int_(np.arange(1, symIdx))] = np.array([natFreq**2 * 1/((natFreq**2 -\
    i**2) + 2*cmath.sqrt(-1) * damping * i * natFreq) for i in freq[1:symIdx]])
    if nPts%2 != 0:
        H[np.int_(np.arange(len(H)-symIdx+1, len(H)))] = \
        np.flipud(np.conj(H[np.int_(np.arange(1, symIdx))]))
    else:
        H[np.int_(np.arange(len(H)-symIdx+2, len(H)))] = \
        np.flipud(np.conj(H[np.int_(np.arange(1, symIdx-1))]))
    sa = max(abs(np.real(np.fft.ifft(np.multiply(H, fas)))))
    return {'PGA':PGA, 'sa':sa}

damping = 0.05
outputsDir = directory.parents[1] / ".applications/case1/Output1"
gm_path = directory.parents[1]/'RCMRF/sample/groundMotion'
dt_file = np.array(pd.read_csv(gm_path/'GMR_dts.txt',header=None)[0])
gm_file = list(pd.read_csv(gm_path/'GMR_names1.txt',header=None)[0]) + \
list(pd.read_csv(gm_path/'GMR_names2.txt',header=None)[0])

RS = {}
T = np.arange(0, 4.01, 0.01)
RS['T1'] = T
for i in range(len(dt_file)):
   print(gm_file[i])
   acc = np.array(pd.read_csv(gm_path/gm_file[i],header=None)[0])*4
   dt = dt_file[i]
   Sa = np.zeros(len(T))
   for j in range(len(T)):
       Sa[j] = _responseSpectrum(acc, dt, T[j], 0.05)['sa']
   RS[int(i+1)] = Sa

RS = pd.DataFrame.from_dict(RS)
plt.plot(T, Sa, color = 'k', label = 'Mean GM')
plt.ylabel(r'$\mathregular{S_a}$ [g]', fontsize = 16)
plt.xlabel('T [s]', fontsize = 16)
plt.yticks(np.arange(0.0, 1.6 + 0.2, 0.2))
plt.xticks(np.arange(0.0, 4.0 + 0.5, 0.5))
plt.ylim(0.0, 1.6)
plt.xlim(0.0, 4.0)
plt.legend(loc = 'best', frameon = False, fontsize = 14) 
plt.grid()

# Storing the RS dataframe into a csv file
pd.DataFrame.to_csv(RS,outputsDir/'RS.csv')
RS.to_pickle(outputsDir/'RS.pickle')

##### ----- Alternative approach (GetSaT) <- not suggested
#T1 = .7
## Newmark average acceleration time integration scheme
#gamma = .5
#beta = .25
#ms = 1. # mass
#
#SaT1 = np.array([])
##for rec in range(1):
#for rec in range(len(dt_file)):
#    print(gm_file[rec])
#    dt = dt_file[rec]
#    acc = np.array(pd.read_csv(gm_path/gm_file[rec],header=None)[0])*9.81
#    p = -ms*acc
#    
#    # calculate initial values
#    k = ms*np.power(2*np.pi/T1,2)
#    w = np.power(k/ms,0.5)
#    c = 2*damping*ms*w
#    a0 = p[0]/ms
#    k_bar = k + gamma*c/beta/dt + ms/beta/dt**2
#    A = ms/beta/dt + gamma*c/beta
#    B = ms/2/beta + dt*c*(gamma/2/beta-1)
#    
#    # Initialization
#    u = np.array([0])
#    v = np.array([0])
#    a = np.array([a0])
#    du = np.array([])
#    dv = np.array([])
#    da = np.array([])
#    dp = np.array([])
#    dp_bar = np.array([])
#    
#    for i in range(1,len(acc)):
#        dp = np.append(dp,p[i]-p[i-1])
#        dp_bar = np.append(dp_bar,dp[i-1] + A*v[i-1] + B*a[i-1])
#        du = np.append(du,dp_bar[i-1]/k_bar)
#        dv = np.append(dv,gamma*du[i-1]/beta/dt - gamma*v[i-1]/beta + dt*(1-gamma/2/beta)*a[i-1])
#        da = np.append(da,du[i-1]/beta/dt**2 - v[i-1]/beta/dt - a[i-1]/2/beta)
#        u = np.append(u,u[i-1] + du[i-1])
#        v = np.append(v,v[i-1] + dv[i-1])
#        a = np.append(a,a[i-1] + da[i-1])
#    
#    umax = 0.
#    for i in range(1,len(acc)):
#        temp1 = abs(u[i-1])
#        if temp1 > umax: umax = temp1
#        
#    pga = 0.
#    for i in range(1,len(acc)):
#        temp1 = abs(acc[i-1])
#        if temp1 > pga: pga = temp1
#    
#    # Calculate spectral values
#    Sd = umax
#    Sv = Sd*w
#    Sa = Sd*w**2/9.81
#    SaT1 = np.append(SaT1,Sa)

#RS = pd.read_pickle('LAquilaRS.pickle')

# Function for timing
def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

# --------- Stop the clock and report the time taken in seconds
elapsed = timeit.default_timer() - start_time           
print('Running time: ',truncate(elapsed,1), ' seconds')
print('Running time: ',truncate(elapsed/float(60),2), ' minutes')
