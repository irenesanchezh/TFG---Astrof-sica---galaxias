import numpy as np 
import bagpipes as pipes
import pymultinest as pmn
from matplotlib import pyplot as plt
import pandas as pd


from astropy.io import fits

def load_data(ID):
    """ Load UltraVISTA photometry from catalogue. """
    
    # Load up the relevant columns from the catalogue.
    cat = np.loadtxt("fot.csv", usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31))
    
    
    # Find the correct row for the object we want.
    row = int(ID) - 1

    # Extract the object we want from the catalogue.
    fluxes = cat[row, 6:19]
    fluxerrs = cat[row, 19:]
    coordenadas = cat[row, :2] 

    # Turn these into a 2D array.
    photometry = np.c_[fluxes, fluxerrs]

    return photometry

    # blow up the errors associated with any missing fluxes.
    for i in range(len(photometry)):
        if (photometry[i, 0] == 0.) or (photometry[i, 1] <= 0):
            photometry[i,:] = [0., 9.9*10**99.]
            
    # Enforce a maximum SNR of 20, or 10 in the IRAC channels.
    for i in range(len(photometry)):
        if i < 10:
            max_snr = 20.
            
        else:
            max_snr = 10.
        
        if photometry[i, 0]/photometry[i, 1] > max_snr:
            photometry[i, 1] = photometry[i, 0]/max_snr

    return photometry

eg_filt_list = ["SLOAN_SDSS.u.dat", "SLOAN_SDSS.g.dat", "SLOAN_SDSS.r.dat", "SLOAN_SDSS.i.dat", "SLOAN_SDSS.z.dat",
                "SLOAN_SDSS.g.dat", "SLOAN_SDSS.i.dat", "SLOAN_SDSS.r.dat", "SLOAN_SDSS.z.dat",
                "WISE_WISE.W1.dat", "WISE_WISE.W2.dat", "WISE_WISE.W3.dat", "WISE_WISE.W4.dat"]

exp = {}                                  # Tau-model star-formation history component
exp["age"] = (0.1, 15.)                   # Vary age between 100 Myr and 15 Gyr. In practice 
                                          # the code automatically limits this to the age of
                                          # the Universe at the observed redshift.

exp["tau"] = (0.3, 10.)                   # Vary tau between 300 Myr and 10 Gyr
exp["massformed"] = (1., 15.)             # vary log_10(M*/M_solar) between 1 and 15
exp["metallicity"] = (0., 2.5)            # vary Z between 0 and 2.5 Z_oldsolar

dust = {}                                 # Dust component
dust["type"] = "Calzetti"                 # Define the shape of the attenuation curve
dust["Av"] = (0., 2.)                     # Vary Av between 0 and 2 magnitudes

fit_instructions = {}                     # The fit instructions dictionary
fit_instructions["redshift"] = (0., 10.)  # Vary observed redshift from 0 to 10
fit_instructions["exponential"] = exp   
fit_instructions["dust"] = dust

nobj = 112

def calcular_propiedades_galaxias(nobj):
    redshift = np.zeros((3, nobj - 1))
    smass = np.zeros((3, nobj - 1))
    sfr = np.zeros((3, nobj - 1))
    age = np.zeros((3, nobj - 1))
    mformed = np.zeros((3, nobj - 1))

    for x in range(1, nobj):
        galaxy = pipes.galaxy(str(x), load_data, filt_list=eg_filt_list, spectrum_exists=False)
        fit = pipes.fit(galaxy, fit_instructions)
        fit.fit(verbose=False)

        np.percentile(fit.posterior.samples["redshift"], (16, 50, 84), out=redshift[:, x - 1])
        np.percentile(fit.posterior.samples["stellar_mass"], (16, 50, 84), out=smass[:, x - 1])
        np.percentile(fit.posterior.samples["sfr"], (16, 50, 84), out=sfr[:, x - 1])
        np.percentile(fit.posterior.samples["exponential:age"], (16, 50, 84), out=age[:, x - 1])
        np.percentile(fit.posterior.samples["exponential:massformed"], (16, 50, 84), out=mformed[:, x - 1])

    return redshift, smass, sfr, age, mformed

redshift, smass, sfr, age, mformed = calcular_propiedades_galaxias(nobj)

# Cargar el archivo fot.csv
df = pd.read_csv("fot.csv")
median_redshift_list = []

nobj = len(df) + 1 
# Bucle para calcular median_redshift y guardar los valores en la lista
for x in range(1, nobj):
    galaxy = pipes.galaxy(str(x), load_data, filt_list=eg_filt_list, spectrum_exists=False)
    fit = pipes.fit(galaxy, fit_instructions)
    fit.fit(verbose=False)
    
    median_redshift_val = np.percentile(fit.posterior.samples["redshift"], 50)
    median_redshift_list.append(median_redshift_val)


median_redshift_array = np.array(median_redshift_list)
df_filtered = df[(median_redshift_array >= 0.2) & (median_redshift_array <= 0.4)]
#df_filtered.to_csv("fot_filtered.csv", index=False)

#Deefino low, median y high percentiles para poder calcular los errores de cada propiedad en los plots:
low_redshift = redshift[0,:]
median_redshift = redshift[1,:]
high_redshift = redshift[2,:]   

low_smass = smass[0,:]
median_smass = smass[1,:]
high_smass = smass[2,:]

low_sfr = sfr[0,:]
median_sfr = sfr[1,:]
high_sfr = sfr[2,:]

low_mformed = mformed[0,:]
median_mformed = mformed[1,:]
high_mformed = mformed[2,:]

low_age = age[0,:]
median_age = age[1,:]
high_age = age[2,:]

    
# PLOTS
# Filtrar los percentiles según el rango de redshift deseado
redshift_in_range = (median_redshift >= 0.2) & (median_redshift <= 0.4)

# Redshift
low_redshift_in_range = low_redshift[redshift_in_range]
median_redshift_in_range = median_redshift[redshift_in_range]
high_redshift_in_range = high_redshift[redshift_in_range]

# Masa estelar
low_smass_in_range = low_smass[redshift_in_range]
median_smass_in_range = median_smass[redshift_in_range]
high_smass_in_range = high_smass[redshift_in_range]

# SFR
low_sfr_in_range = low_sfr[redshift_in_range]
median_sfr_in_range = median_sfr[redshift_in_range]
high_sfr_in_range = high_sfr[redshift_in_range]

# Masa formada
low_mformed_in_range = low_mformed[redshift_in_range]
median_mformed_in_range = median_mformed[redshift_in_range]
high_mformed_in_range = high_mformed[redshift_in_range]

# Edad
low_age_in_range = low_age[redshift_in_range]
median_age_in_range = median_age[redshift_in_range]
high_age_in_range = high_age[redshift_in_range]

# PLOTS
# Histograma redshift
plt.hist(median_redshift, bins=np.arange(0, 1, 0.05))
plt.title("Redshift de mi cúmulo")
plt.xlabel("Redshift")
plt.ylabel("Número de galaxias")
#plt.show()

# Histograma masa estelar
plt.hist(median_smass, bins=np.arange(8.5, 13, 0.1))
plt.title("Masa estelar de las galaxias de mi cúmulo")
plt.xlabel("Masa estelar")
plt.ylabel("Número de galaxias")
#plt.show()


# Masa estelar frente a redshift
plt.xlim(0., 1.)
plt.errorbar(median_redshift_in_range, median_smass_in_range,
             xerr=np.array(list(zip(median_redshift_in_range - low_redshift_in_range, high_redshift_in_range - median_redshift_in_range))).T,
             yerr=np.array(list(zip(median_smass_in_range - low_smass_in_range, high_smass_in_range - median_smass_in_range))).T,
             ecolor='red', fmt='.', label="Redshift entre 0.2 y 0.4")

plt.errorbar(median_redshift[~redshift_in_range], median_smass[~redshift_in_range],
             xerr=np.array(list(zip(median_redshift[~redshift_in_range] - low_redshift[~redshift_in_range], high_redshift[~redshift_in_range] - median_redshift[~redshift_in_range]))).T,
             yerr=np.array(list(zip(median_smass[~redshift_in_range] - low_smass[~redshift_in_range], high_smass[~redshift_in_range] - median_smass[~redshift_in_range]))).T,
             ecolor='blue', fmt='.', label="Redshift fuera de 0.2-0.4")

plt.title("Masa estelar vs. redshift")
plt.xlabel("z")
plt.ylabel("Masa estelar (masas solares)")
plt.legend()
#plt.show()

# Masa estelar vs. log(SFR)
plt.errorbar(median_smass_in_range, np.log10(median_sfr_in_range),
             xerr=np.array(list(zip(median_smass_in_range - low_smass_in_range, high_smass_in_range - median_smass_in_range))).T,
             yerr=np.array(list(zip(np.log10(median_sfr_in_range) - np.log10(low_sfr_in_range), np.log10(high_sfr_in_range) - np.log10(median_sfr_in_range)))).T,
             ecolor='red', fmt='.', label="Redshift entre 0.2 y 0.4")

plt.errorbar(median_smass[~redshift_in_range], np.log10(median_sfr[~redshift_in_range]),
             xerr=np.array(list(zip(median_smass[~redshift_in_range] - low_smass[~redshift_in_range], high_smass[~redshift_in_range] - median_smass[~redshift_in_range]))).T,
             yerr=np.array(list(zip(np.log10(median_sfr[~redshift_in_range]) - np.log10(low_sfr[~redshift_in_range]), np.log10(high_sfr[~redshift_in_range]) - np.log10(median_sfr[~redshift_in_range])))).T,
             ecolor='blue', fmt='.', label="Redshift fuera de 0.2-0.4")

plt.title("log(Masa estelar) vs. log(SFR)")
plt.xlabel("log(Masa estelar) (log masas solares)")
plt.ylabel("log(SFR) (log masas solares/año)")
plt.legend()
#plt.show()

# Masa estelar vs. log(edad estelar)
plt.errorbar(median_smass_in_range, np.log10(median_age_in_range),
             xerr=np.array(list(zip(median_smass_in_range - low_smass_in_range, high_smass_in_range - median_smass_in_range))).T,
             yerr=np.array(list(zip(np.log10(median_age_in_range) - np.log10(low_age_in_range), np.log10(high_age_in_range) - np.log10(median_age_in_range)))).T,
             ecolor='red', fmt='.', label="Redshift entre 0.2 y 0.4")

plt.errorbar(median_smass[~redshift_in_range], np.log10(median_age[~redshift_in_range]),
             xerr=np.array(list(zip(median_smass[~redshift_in_range] - low_smass[~redshift_in_range], high_smass[~redshift_in_range] - median_smass[~redshift_in_range]))).T,
             yerr=np.array(list(zip(np.log10(median_age[~redshift_in_range]) - np.log10(low_age[~redshift_in_range]), np.log10(high_age[~redshift_in_range]) - np.log10(median_age[~redshift_in_range])))).T,
             ecolor='blue', fmt='.', label="Redshift fuera de 0.2-0.4")

plt.title("log(Masa estelar) vs. log(edad estelar)")
plt.xlabel("log(Masa estelar) (masas solares)")
plt.ylabel("log(edad estelar) (unidades)")
plt.legend()
#plt.show()

#Histograma distancia al centro del cúmulo

ra_centro = 328.4  #deg      #Coordenadas centro del cúmulo
dec_centro = 17.695  #deg
ra_galaxias = coordenadas[:, 0]  #Coordenadas (ra, dec) de cada galaxia
dec_galaxias = coordenadas[:, 1]  
ra_centro_rad = np.radians(ra_centro)   #Paso a radianes
dec_centro_rad = np.radians(dec_centro)
ra_galaxias_rad = np.radians(ra_galaxias)
dec_galaxias_rad = np.radians(dec_galaxias)
delta_ra = ra_galaxias_rad - ra_centro_rad
delta_dec = dec_galaxias_rad - dec_centro_rad
distancias_angulares = np.arccos(np.sin(dec_centro_rad) * np.sin(dec_galaxias_rad) + np.cos(dec_centro_rad) * np.cos(dec_galaxias_rad) * np.cos(delta_ra))
distancias_lineales = np.degrees(distancias_angulares) #Distancia angular -> distancia lineal (grados)

plt.hist(median_redshift, bins=20)
plt.title("Distancia al centro del cúmulo")
plt.xlabel("Distancia (grados)")
plt.ylabel("Número de galaxias")
plt.show()


#print (median_redshift)

#fit.fit(verbose=True)
#fit.plot_spectrum_posterior(save=False, show=True)
#fit.plot_sfh_posterior(save=False, show=True)
#fit.plot_corner(save=False, show=True)
