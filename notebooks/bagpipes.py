import numpy as np 
import bagpipes as pipes
import pymultinest as pmn

from astropy.io import fits

def load_data(ID):
    """ Load UltraVISTA photometry from catalogue. """

    # Load up the relevant columns from the catalogue.
    cat = np.loadtxt("fot_filtered.csv", usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31))
    
    # Find the correct row for the object we want.
    row = int(ID) - 1

    # Extract the object we want from the catalogue.
    fluxes = cat[row, 6:19]
    fluxerrs = cat[row, 19:]

    # Turn these into a 2D array.
    photometry = np.c_[fluxes, fluxerrs]

    return photometry

    # blow up the errors associated with any missing fluxes.
    for i in range(len(photometry)):
        if (photometry[i, 0] == 0.) or (photometry[i, 1] <= 0):
            photometry[i,:] = [0., 9.9*10**99.]
            
    return photometry

eg_filt_list = ["SLOAN_SDSS.u.dat", "SLOAN_SDSS.g.dat", "SLOAN_SDSS.r.dat", "SLOAN_SDSS.i.dat", "SLOAN_SDSS.z.dat",
                "SLOAN_SDSS.g.dat", "SLOAN_SDSS.i.dat", "SLOAN_SDSS.r.dat", "SLOAN_SDSS.z.dat",
                "WISE_WISE.W1.dat", "WISE_WISE.W2.dat", "WISE_WISE.W3.dat", "WISE_WISE.W4.dat"]

num_lines = sum(1 for _ in open('fot_filtered.csv'))
print (num_lines)

for i in range(0,num_lines):
    galid = str(i)
    galaxy = pipes.galaxy(galid, load_data, filt_list=eg_filt_list, spectrum_exists=False)
    exp = {}                                  # Tau-model star-formation history component
    exp["age"] = (0.1, 15.)                   # Vary age between 100 Myr and 15 Gyr 
    exp["tau"] = (0.3, 10.)                   # Vary tau between 300 Myr and 10 Gyr
    exp["massformed"] = (1., 15.)             # vary log_10(M*/M_solar) between 1 and 15
    exp["metallicity"] = (0., 2.5)            # vary Z between 0 and 2.5 Z_oldsolar
    dust = {}                                 # Dust component
    dust["type"] = "Calzetti"                 # Define the shape of the attenuation curve
    dust["Av"] = (0., 2.)                     # Vary Av between 0 and 2 magnitudes
    fit_instructions = {}                     # The fit instructions dictionary
    fit_instructions["redshift"] = 0.23       # Vary observed redshift from 0 to 10
    fit_instructions["exponential"] = exp   
    fit_instructions["dust"] = dust
    fit = pipes.fit(galaxy, fit_instructions, run='posterior_redshift_fixed_irene')
    fit.fit(verbose=True)

