from __future__ import division
import pandas as pd
import cmath
import numpy as np
import os
import matplotlib.pyplot as plt

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


sp = os.path.dirname(os.path.realpath(__file__))
data = pd.read_table('spectrum.txt', header = None)
plt.figure(figsize=(10, 5))  
plt.plot(data[0], data[1]/9.81, color = 'r', label = 'EC8-1')
sa_dict = {}
sp = os.path.dirname(os.path.realpath(__file__))
for file in os.listdir(sp):
    if file.endswith('.DAT'):
        record = file
        if record == '00122L.DAT':
            sf = 5.1107
        if record == '00138L.DAT':
            sf = 4.1688       
        if record == '00293T.DAT':
            sf = 9.6488
        if record == '01144L.DAT':
            sf = 4.4760
        if record == '01155L.DAT':
            sf = 3.9949
        if record == '01163L.DAT':
            sf = 4.1182
        if record == '01177L.DAT':
            sf = 4.1607
        data = pd.read_csv(record, header = None, delim_whitespace=True)
        acc_list = data[1]*sf
        npts = len(acc_list)
        dt = data[0][0]
        T_list = []
        Sa_list = []
        for T in np.arange(0, 4.01, 0.01):
            Sa = _responseSpectrum(acc_list, dt, T, 0.05)['sa']
            T_list.append(T)
            Sa_list.append(Sa)                                 
        plt.plot(T_list, Sa_list, color = '0.8')
        sa_dict[record] = Sa_list
sa_sum = 0
for record in sa_dict:
    print(np.array(sa_dict[record]))
    sa_sum = sa_sum + np.array(sa_dict[record])
sa_av = sa_sum/len(sa_dict)
plt.plot(T_list, sa_av, color = 'k', label = 'Mean GM')
plt.ylabel(r'$\mathregular{S_a}$ [g]', fontsize = 16)
plt.xlabel('T [s]', fontsize = 16)
plt.yticks(np.arange(0.0, 1.6 + 0.2, 0.2))
plt.xticks(np.arange(0.0, 4.0 + 0.5, 0.5))
plt.ylim(0.0, 1.6)
plt.xlim(0.0, 4.0)
plt.legend(loc = 'best', frameon = False, fontsize = 14) 
plt.grid()    
plt.savefig('ScaledGMs.tiff', dpi = 200, bbox_inches='tight')                  