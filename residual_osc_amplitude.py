import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def residual_amplitude(teff, logg, mstar, rstar, exptime, readtime=0, nobs=1,
                       nu_max_sun=3100, delta_nu_sun=134.9, a_max_sun=0.19, c_env_sun=331, 
                       S=[1.0,1.35,1.02,0.47], logg_sun=4.44, teff_sun=5777, method='chaplin', nmodes=13):
    """
    Parameters
    ----------
    nu : float or array*
        Frequency at which to calculate transfer function (microHz)
    nu_exp : float or array*
        Exposure time in frequency space (microHz)
    nu_max : float or array*
        Central frequency of oscillation power envelope (microHz)
    a_max : float or array*
        Oscillation amplitude at nu_max (m/s)
    c_env : float or array*
        Characteristic width of oscillation envelope (microHz)
    S : float, array, or list of floats
        Spatial response functions
    mstar : float or array*
        Stellar mass (solar units); used to calculate frequency spacing
    rstar : float or array*
        Stellar radius (solar units); used to calculate frequency spacing 
    
    readtime : float, optional
        Readout time (seconds)  for sequences of exposures. The default is 0.
    nobs : float, optional
        Number of exposures in a sequence. The default is 1.
        
    optional solar constants: nu_max_sun, delta_nu_sun, a_max_sun, c_env_sun,
                              logg_sun, teff_sun
    
    method : string, ['chaplin', 'comb']
        Method by which residual amplitude is calculated.
        For 'chaplin' we integrate over the filtered oscillation power excess
        following the methods of Chaplin et al. (2019).
        
        For 'comb' we instead generate a comb of discrete oscillation modes
        and sum over the filtered power of these modes.
        The default is 'chaplin'.
        
        *all input parameters must be floats, not arrays, for chaplin method
        
    nmodes : optional, int
        Number of oscillation modes per l order to use for the comb method.
        The default is 13.



    Returns
    -------
    residual amplitude : float or array
        Filtered power spectral density at input frequency
    """
    
    nu_exp=1.0e6/exptime # exposure time in microHz

    nu_max=nu_max_sun*10**(logg-logg_sun)/np.sqrt(teff/teff_sun) # Equation 6 of Chaplin et al. (2019)

    a_max=a_max_sun*(teff/teff_sun)**4/10**(logg-logg_sun) # Equation 7 of Chaplin et al. (2019)
    
    if teff<teff_sun:
        c_env=c_env_sun*(nu_max/nu_max_sun)**0.88 # Equation 8 of Chaplin et al. (2019)
    else:
        c_env=c_env_sun*(nu_max/nu_max_sun)**0.88*(1+(teff-teff_sun)/1667.) # Equation 8 of Chaplin et al. (2019)
    
    delta_nu=delta_nu_sun*np.sqrt(mstar/rstar**3)
    
    if method=='chaplin':
        
        # choose some range over which to integrate
        # defaulting to 6x delta_nu, centered on nu_max
        
        n=float(int(nu_max/c_env)-3)
        minfreq=nu_max-n*c_env
        maxfreq=nu_max+n*c_env
        
        integral=quad(filtered_psd, minfreq, maxfreq, args=(nu_exp, nu_max, a_max, c_env, S, mstar, rstar, readtime, nobs))[0]
    
    elif method=='comb':
        
        nu=np.arange(-1*int(nmodes/2),int(nmodes/2)+1,1)*delta_nu+nu_max

        # approximate the small frequency spacing as delta_nu/10
        # this is reasonably accurate for a Sun-like star
        
        nu_l0=nu[:-1]+delta_nu/2.
        nu_l1=nu
        nu_l2=nu[:-1]+delta_nu/2.-delta_nu/10.
        nu_l3=nu-delta_nu/10.


        integral=0
        for n in range(len(nu)):
            integral+=filtered_psd(nu_l1[n], nu_exp, nu_max, a_max, c_env, S, mstar, rstar, readtime, nobs)*delta_nu*S[1]**2/np.sum(np.array(S)**2)
            integral+=filtered_psd(nu_l3[n], nu_exp, nu_max, a_max, c_env, S, mstar, rstar, readtime, nobs)*delta_nu*S[3]**2/np.sum(np.array(S)**2)
        for n in range(len(nu)-1):
            integral+=filtered_psd(nu_l0[n], nu_exp, nu_max, a_max, c_env, S, mstar, rstar, readtime, nobs)*delta_nu*S[0]**2/np.sum(np.array(S)**2)
            integral+=filtered_psd(nu_l2[n], nu_exp, nu_max, a_max, c_env, S, mstar, rstar, readtime, nobs)*delta_nu*S[2]**2/np.sum(np.array(S)**2)

    residual = np.sqrt(integral)

    return residual

def plot_residual_amplitudes(exptimes, residuals, idx_prec=None, idx_minus=None, idx_plus=None, logx=False, logy=True, levels=[], title=None):
    fig, ax = plt.subplots(figsize=(6,5))
    ax.tick_params(length=6, width=1.5, axis='both', labelsize=10, direction='out', color='#383838', colors='#383838')
    ax.tick_params(length=4, which='minor', width=1, axis='both', labelsize=7, direction='out', color='#383838', colors='#383838')
    for xx in ['top','bottom','left','right']:
        ax.spines[xx].set_linewidth(1.5)
        ax.spines[xx].set_color('#383838')

    ax.plot(exptimes/60., residuals, color='k', lw=1.5, zorder=0)

    
    if idx_prec:
        ax.axvline(x=exptimes[idx_prec]/60., ls='--', color='C7', alpha=0.7, lw=1.5, zorder=-1)
        ax.plot(np.array([exptimes[idx_prec]])/60., [residuals[idx_prec]], marker='o', ls='', mfc='r', mec='k', mew=2, ms=8, zorder=1)
    if idx_minus:
        ax.axvline(x=exptimes[idx_minus]/60., ls='--', color='C7', alpha=0.7, lw=1.5, zorder=-1)
        ax.plot(np.array([exptimes[idx_minus]])/60., [residuals[idx_minus]], marker='o', ls='', mfc='w', mec='k', mew=2, ms=8, zorder=1)
    if idx_plus:
        ax.axvline(x=exptimes[idx_plus]/60., ls='--', color='C7', alpha=0.7, lw=1.5, zorder=-1)
        ax.plot(np.array([exptimes[idx_plus]])/60., [residuals[idx_plus]], marker='o', ls='', mfc='w', mec='k', mew=2, ms=8, zorder=1)
    
    for level in levels:
        ax.axhline(y=level, ls='-', color='C7', alpha=0.7, lw=0.5, zorder=-1)
        if logx:
            ax.text((10**(0.85*np.log10(exptimes[-1])+0.15*np.log10(exptimes[0])))/60.,level*1.05,"%d cm/s" % (level*100), fontsize=10, color='#383838')
        else:
            ax.text((0.85*exptimes[-1]+0.15*exptimes[0])/60.,level*1.05,"%d cm/s" % (level*100), fontsize=10, color='#383838')
            
    if logy:
        ax.set_yscale('log')
    if logx:
        ax.set_xscale('log')

    ax.set_xlim(exptimes[0]/60., exptimes[-1]/60.)
    ax.set_ylim(residuals[-1], residuals[0])

    ax.set_xlabel('Integration Time [min]', fontsize=14)
    ax.set_ylabel('Residual Amplitude [m s$^{-1}$]', fontsize=14)

    if title:
        ax.set_title(title, fontsize=14)

    plt.show()

def predict_exptime(teff, logg, mstar, rstar, precision,
                    nu_max_sun=3100, delta_nu_sun=134.9, a_max_sun=0.19, c_env_sun=331, 
                    S=[1.0,1.35,1.02,0.47], logg_sun=4.44, teff_sun=5777, method='chaplin', nmodes=13,
                    show_plot=True):
    """
    Calculates and reports the expected exposure time required to reach the
    requested residual amplitude (precision; float)

    Returns the exposure time to reach this level, as well as the exposure times
    to reach a 10% worse precision and a 10% better precision
    """
    
    ncoarse=25
    coarse_exptimes=np.logspace(1.5,4.0,ncoarse)
    coarse_residuals=np.zeros(ncoarse)
    for n in range(ncoarse):
        coarse_residuals[n]=residual_amplitude(teff, logg, mstar, rstar, coarse_exptimes[n],
                                               nu_max_sun=nu_max_sun, delta_nu_sun=delta_nu_sun, a_max_sun=a_max_sun, c_env_sun=c_env_sun,
                                               S=S, logg_sun=logg_sun, teff_sun=teff_sun, method=method, nmodes=nmodes)
    idx_start=np.where(coarse_residuals>5.*precision)[0]
    if len(idx_start)>0:
        t_search_start=coarse_exptimes[idx_start[-1]]
    else:
        t_search_start=coarse_exptimes[0]
    idx_end=np.where(coarse_residuals<0.8*precision)[0]
    if len(idx_end)>0:
        t_search_end=coarse_exptimes[idx_end[0]]
    else:
        t_search_end=coarse_exptimes[-1]
    
    nfine=250
    fine_exptimes=np.logspace(np.log10(t_search_start),np.log10(t_search_end),nfine)
    fine_residuals=np.zeros(nfine)
    for m in range(nfine):
        fine_residuals[m]=residual_amplitude(teff, logg, mstar, rstar, fine_exptimes[m],
                                               nu_max_sun=nu_max_sun, delta_nu_sun=delta_nu_sun, a_max_sun=a_max_sun, c_env_sun=c_env_sun,
                                               S=S, logg_sun=logg_sun, teff_sun=teff_sun, method=method, nmodes=nmodes)
    
    idx_plus=np.where(fine_residuals<precision*1.1)[0][0]
    idx_prec=np.where(fine_residuals<precision)[0][0]
    idx_minus=np.where(fine_residuals<precision*0.9)[0][0]
    
    t_plus10=fine_exptimes[np.where(fine_residuals<precision*1.1)[0][0]]
    t_prec=fine_exptimes[np.where(fine_residuals<precision)[0][0]]
    t_minus10=fine_exptimes[np.where(fine_residuals<precision*0.9)[0][0]]
    
    if show_plot:
        plot_residual_amplitudes(fine_exptimes,fine_residuals,idx_prec=idx_prec,
                          idx_minus=idx_minus,idx_plus=idx_plus)
        
    return t_prec, np.array([t_plus10, t_minus10])

def calc_delta_nu(mstar, rstar, delta_nu_sun=134.9):
    """
    Parameters
    ----------
    mstar : float or array
        Stellar mass (solar units)
    rstar : float or array
        Stellar radius (solar units)
    delta_nu_sun : float, optional
        Solar delta nu value (microHz). The default is 134.9 microHz.

    Returns
    -------
    delta_nu : float or array
        Large frequency spacing (microHz)
    """

    return delta_nu_sun*np.sqrt(mstar/rstar**3)


def filtered_psd(nu, nu_exp, nu_max, a_max, c_env, S, mstar, rstar, readtime=0, nobs=1):   
    """
    Parameters
    ----------
    nu : float or array
        Frequency at which to calculate transfer function (microHz)
    nu_exp : float or array
        Exposure time in frequency space (microHz)
    nu_max : float or array
        Central frequency of oscillation power envelope (microHz)
    a_max : float or array
        Oscillation amplitude at nu_max (m/s)
    c_env : float or array
        Characteristic width of oscillation envelope (microHz)
    S : float, array, or list of floats
        Spatial response functions
    mstar : float or array
        Stellar mass (solar units); used to calculate frequency spacing
    rstar : float or array
        Stellar radius (solar units); used to calculate frequency spacing
    
    readtime : float, optional
        Readout time (seconds)  for sequences of exposures. The default is 0.
    nobs : float, optional
        Number of exposures in a sequence. The default is 1.

    Returns
    -------
    filtered_psd : float or array
        Filtered power spectral density at input frequency
    """

    if readtime==0:
        """
        if a single continuous exposure, transfer function is simply a sinc;
        use Equation 2 of Chaplin et al. (2019)
        """
        eta=np.abs(np.sinc(nu/nu_exp))
    else:
        """
        if a sequence of exposures, transfer function is more complicated;
        use Equation 23 of Gupta et al. (2022)
        """
        exptime=1e6/nu_exp #convert from microHz to seconds
        eta_c=0
        for m in range(nobs):
            eta_c+=np.exp((2*m-nobs+1)*1j*np.pi*nu/1e6*exptime)*np.exp((2*m-nobs+1)*1j*np.pi*nu/1e6*readtime)
        eta=np.sin(np.pi*nu/1e6*exptime)/(np.pi*nu/1e6*(exptime*nobs))*np.abs(eta_c.real)

    power = a_max**2*np.sum(np.array(S)**2)*np.exp(-0.5*(nu-nu_max)**2/c_env**2)

    filtered_power = eta**2 * power

    delta_nu = calc_delta_nu(mstar, rstar)
    
    return filtered_power / delta_nu
    
if __name__ == "__main__":
    
    import argparse
    import os
    import pandas as pd
    
    parser = argparse.ArgumentParser(
                    prog = 'Osc Exptime',
                    description = 'Exposure time estimator for oscillation filtering')
    
    parser.add_argument('starname', type=str)
    parser.add_argument('precision', type=float)
    parser.add_argument('-v', '--verbose', action='store_true', default=True)
    parser.add_argument('-p', '--plot', action='store_true', default=True)
    parser.add_argument('-s', '--save', action='store_true', default=False)
    parser.add_argument('-f', '--filename', default='star_inputs.csv')
    args = parser.parse_args()

    inputfile=pd.read_csv(os.path.join('inputs',args.filename))
        
    if args.starname!='ALL':
        teff=np.array(inputfile.teff[inputfile.starname==args.starname])[0]
        logg=np.array(inputfile.logg[inputfile.starname==args.starname])[0]
        mstar=np.array(inputfile.mstar[inputfile.starname==args.starname])[0]
        rstar=np.array(inputfile.rstar[inputfile.starname==args.starname])[0]
                
        t_prec, tpm = predict_exptime(teff, logg, mstar, rstar, args.precision, show_plot=args.plot)
        
        if args.verbose:
            print('---------')
            print(args.starname)
            print('---------')
            print("Time to reach %.3f m/s residual oscillation amplitude: %d seconds\n" % (args.precision, t_prec))
            print("Time to reach %.3f m/s residual oscillation amplitude: %d seconds" % (args.precision*1.1, tpm[0]))
            print("(%d seconds [%.1f%%] FASTER than time to requested precision)\n" % (t_prec-tpm[0], 100.*(t_prec-tpm[0])/t_prec))
            print("Time to reach %.3f m/s residual oscillation amplitude: %d seconds" % (args.precision*0.9, tpm[1]))
            print("(%d seconds [%.1f%%] SLOWER than time to requested precision)" % (tpm[1]-t_prec, 100.*(tpm[1]-t_prec)/t_prec))
            print('---------')
        
    elif args.starname=='ALL':
        
        t_prec_save=np.zeros(len(np.unique(inputfile.starname)))
        
        for s, starname in enumerate(np.unique(inputfile.starname)):
            teff=np.array(inputfile.teff[inputfile.starname==starname])[0]
            logg=np.array(inputfile.logg[inputfile.starname==starname])[0]
            mstar=np.array(inputfile.mstar[inputfile.starname==starname])[0]
            rstar=np.array(inputfile.rstar[inputfile.starname==starname])[0]
    
            t_prec, tpm = predict_exptime(teff, logg, mstar, rstar, args.precision, show_plot=False)

            if args.verbose:
                print('---------')
                print(starname)
                print('---------')
                print("Time to reach %.3f m/s residual oscillation amplitude: %d seconds\n" % (args.precision, t_prec))
                print("Time to reach %.3f m/s residual oscillation amplitude: %d seconds" % (args.precision*1.1, tpm[0]))
                print("(%d seconds [%.1f%%] FASTER than time to requested precision)\n" % (t_prec-tpm[0], 100.*(t_prec-tpm[0])/t_prec))
                print("Time to reach %.3f m/s residual oscillation amplitude: %d seconds" % (args.precision*0.9, tpm[1]))
                print("(%d seconds [%.1f%%] SLOWER than time to requested precision)" % (tpm[1]-t_prec, 100.*(tpm[1]-t_prec)/t_prec))
                print('---------')
                
            t_prec_save[s]=t_prec
            
        if args.save==True:
            df=pd.DataFrame()
            df['starname']=np.unique(inputfile.starname)
            df['exptime']=t_prec_save
            df.to_csv(os.path.join('outputs',os.path.splitext(args.filename)[0]+'_'+str(args.precision).replace('.','p')+'.csv'), index=False)