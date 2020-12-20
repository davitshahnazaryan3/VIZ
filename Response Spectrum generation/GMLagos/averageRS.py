"""
1. Get average of all scaled records
2. Get spectral acceleration corresponding to the fundamental period of the structure

"""

# Initializaiton
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

record_dir = os.path.dirname(os.path.realpath(__file__))
spectra_dir = record_dir + "\\" + "Spectra"

# Import RS Database
rs = pd.read_csv('RS.txt')

# Import reference for the datafiles
ref = pd.read_csv('ref_file.txt')

# Initialization of average RS
avg = np.zeros(len(rs['1-ImperialValley-02.dat']))

count  = 0
for record_name in ref['GM']:
    avg += rs[record_name]
    
    count += 1
    
average = avg/count
rs['Average'] = average

avgSpectra = open(spectra_dir + "\\" + "Average_Spectrum" + ".tcl", "w+")
for i in average:
    avgSpectra.write(str(i) + "\n")
avgSpectra.close()

# Plotting
plt.rc('font', family='Garamond') 
plt.rc('font', serif='Garamond')
plt.figure(figsize = (10, 5))

x = np.arange(0.05, 3.01, 0.01)

for record_name in ref['GM']:
    y = rs[record_name]
    
    plt.plot(x, y, color = '0.9', label = '_nolegend_')

y = rs['Average']
plt.plot(x, y, color = 'r', label = 'Average response spectrum')

plt.legend(frameon = False,
         loc = 'upper right',
         fontsize = 12)
plt.grid(color = '0.75', ls = '--', lw = 0.5)

# 4 storey
x_4 = [0.65, 0.65, 0.85, 0.85]
y_4 = [0.00, 1.75, 1.75, 0.00]

plt.annotate('4 storey buildings',
             xy = (0.35, 0.8), 
             xycoords = 'axes fraction',
             xytext = (-100, +30), 
             textcoords = 'offset points', 
             fontsize = 12,
             color = 'b',
             arrowprops = dict(arrowstyle = '-|>', 
                               color = 'b',
                               connectionstyle = 'arc3,rad=.2'))

plt.fill(x_4, y_4, color = 'b', alpha = 0.1)

# 8 storey
x_8 = [0.95, 0.95, 1.10, 1.10]
y_8 = [0.00, 1.75, 1.75, 0.00]

plt.annotate('8 storey buildings',
             xy = (0.5, 0.6), 
             xycoords = 'axes fraction',
             xytext = (+0, +30), 
             textcoords = 'offset points', 
             fontsize = 12,
             color = 'r',
             arrowprops = dict(arrowstyle = '-|>', 
                               color = 'r',
                               connectionstyle = 'arc3,rad=.2'))

plt.fill(x_8, y_8, color = 'r', alpha = 0.1)

plt.xlim(0, 2)
plt.ylim(0, 1.75)
plt.xlabel('Period [s]', fontsize = 14)
plt.ylabel('Spectral acceleration [g]', fontsize = 14)
plt.savefig('RSavg.tiff', dpi = 200, bbox_inches='tight')
plt.show()

# Fundamental period of the structure
T1 = 0.95
T = []
xx = list(x)

for i in range(0,len(xx)):
    T.append(round(float(xx[i]),3))

index_t1 = T.index(T1)

# Spectral acceleration corresponding to the fundamental period of the structure
Sa_T1 =  rs.loc[index_t1,"Average"]



















