import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from residual_osc_amplitude import residual_amplitude, plot_residual_amplitudes


starname='Sun'
use_inputfile=False

if use_inputfile:
    """
    pull stellar inputs from data file
    """
    inputfile=pd.read_csv('star_inputs.csv')
    teff=np.array(inputfile.teff[inputfile.starname==starname])[0]
    logg=np.array(inputfile.logg[inputfile.starname==starname])[0]
    mstar=np.array(inputfile.mstar[inputfile.starname==starname])[0]
    rstar=np.array(inputfile.rstar[inputfile.starname==starname])[0]
    
else:
    """
    manually define stellar parameters
    make sure the stellar mass and radius are consistent with surface gravity
    """
    teff=5777
    logg=4.44
    mstar=1.0
    rstar=1.0

"""
Example calculation for a continuous exposure across a range of exposure times
"""

exptimes=np.logspace(1.5,4.0,250)
residuals=np.zeros(len(exptimes))
for t in range(len(exptimes)):
    residuals[t]=residual_amplitude(teff, logg, mstar, rstar, exptimes[t])
    
plot_residual_amplitudes(exptimes, residuals, logx=True, levels=[0.05,0.1,0.2,0.3], title='Continuous Exposure')


"""
Example calculation for a sequence of exposures
"""

exptime=30
readtime=10
nobs=np.arange(1,100,1)
residuals=np.zeros(len(nobs))
for n in range(len(nobs)):
    residuals[n]=residual_amplitude(teff, logg, mstar, rstar, exptime, readtime=10, nobs=nobs[n])
    
integration_times=nobs*(exptime+readtime)
plot_residual_amplitudes(integration_times, residuals, logx=True, levels=[0.05,0.1,0.2,0.3], title='Sequence of Exposures')
