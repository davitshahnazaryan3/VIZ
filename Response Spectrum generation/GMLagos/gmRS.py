"""


"""

# Initializaiton
import numpy as np
import pandas as pd
import math
import os

# Input Data
ref = pd.read_csv('ref_file.txt')
m = 1. # mass
qsi = 0.05 # damping ratio
g = 1 # ground acceleration [m/s2], 1 = if needs to stay in [g]
n_record = 40 # number of records
record_dir = os.path.dirname(os.path.realpath(__file__))
spectra_dir = record_dir + "\\" + "Spectra"

# Period Interval
T = np.arange(0.05, 3.01, 0.01)
newT = open(spectra_dir + "\\" + "Periods" + ".tcl", "w+")
for i in T:
    newT.write(str(i) + "\n")
newT.close()

# Dataframe of all response spectra
columns = ref["GM"]

rs = pd.DataFrame(index=T, columns=columns)
rs = rs.fillna(0)

for i in range(0, n_record):
    # Load ground motion
    gmName = ref.loc[i,"GM"]
    print(gmName)
    gmotion = open(gmName,"r")
    
    lines = gmotion.readlines()
    f = []
    for x in lines:
        f.append(x.split('  '))
    gmotion.close()
    
    # Transformation from cm/s2 (0.01) or mm/s2 (0.001) into m/s2
    factor = 1.
    
    # Record time step
    dt = round(float(ref.loc[i,"TimeStep"]),4)
    
    # Record scaling factor, comment if no scaling is required
    sf = ref.loc[i,"SF"]
    print(sf)
    # Record number of steps
    nSteps = ref.loc[i,"NSteps"]
    
    gm = []
    for x in range(len(f)):
        # if necessary multiply by sf
        gm.append(float(f[x][1])*factor*g*sf)
    
    # Time interval
    t = []
    for x in range(1, nSteps+1):
        t.append(round((x-1)*dt,4))
    
    ## 1. PGA, PGV, PGD
    accMax = max(gm)
    accMin = min(gm)
    PGA = max(accMax,abs(accMin))
    
    ## Outputting Spectras
    Dsp = []
    Asp = []
    Vsp = []
    w = []
    
    # Initialization of spectrum files
    spectrum = open(spectra_dir + "\\" + "Spectrum_" + gmName + ".tcl", "w+")
    
    for i in range(0, len(T)):
        # Circular frequency
        w.append(2.*math.pi/T[i])
        # Stiffness
        k = m*w[i]**2.
        # Damping
        c = 2.*qsi*math.sqrt(k*m)
        
        d = []
        d.append(0.)
        a0 = -1.*gm[0]
        d.append(d[0] + a0*dt**2/2)
    
        for j in range(1, nSteps, 1):
            d.append(-d[j-1] + 2*d[j] + (dt**2)*(-k*d[j] - m*gm[j] - c*(d[j]-d[j-1])/dt)/(m + c*dt/2))
            
        Dsp.append(max(map(abs,d)))
        Asp.append(w[i]**2*Dsp[i])
        Vsp.append(w[i]*Dsp[i])
        
        spectrum.write(str(w[i]**2*Dsp[i]) + "\n")
        
    rs[gmName] = Asp
    spectrum.close()

#rs.to_pickle(record_dir, compression='infer', protocol=4)




