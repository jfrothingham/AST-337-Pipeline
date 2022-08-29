#!/usr/bin/env python
# coding: utf-8

# In[1]:


# The standard fare, plus a few extra packages:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import os.path
get_ipython().run_line_magic('matplotlib', 'inline')

# Newer packages:
from astropy.stats import mad_std
from astropy.stats import sigma_clip
from photutils.utils import calc_total_error
import astropy.stats as stat
from photutils import aperture_photometry, CircularAperture, CircularAnnulus, DAOStarFinder


# In[ ]:


def bg_error_estimate(fitsfile):
    """
    calculated the back ground error, writes it to a fits image
    Then use the original data and background error to calculated the total error, 
    and write the total error to another fits image.
    """
    fitsdata = fits.getdata(fitsfile)
    hdr = fits.getheader(fitsfile)
    
    # Sigma_clip filters the data and gets rid of outliers that's over or under a specified number of sigma. 
    #(getting rid of stars)
    filtered_data = sigma_clip(fitsdata, sigma=3.,copy=False)
    
    # The values that are outliers will be turned to not a number (nan) (replacing what used to be star pixels with nan)
    # We're taking the square root of the new background image that includes that nan values. 
    # Replacing all the nan values with the median background error 
    bkg_values_nan = filtered_data.filled(fill_value=np.nan)
    bkg_error = np.sqrt(bkg_values_nan)
    bkg_error[np.isnan(bkg_error)] = np.nanmedian(bkg_error)
    
    print("Writing the background-only error image: ", fitsfile.split('.')[0]+"_bgerror.fit")
    fits.writeto(fitsfile.split('.')[0]+"_bgerror.fit", bkg_error, hdr, overwrite=True)
    
    effective_gain = 1.4 # electrons per ADU
    
    # create image from astro photutuils
    error_image = calc_total_error(fitsdata, bkg_error, effective_gain)  
    
    print("Writing the total error image: ", fitsfile.split('.')[0]+"_error.fit")
    fits.writeto(fitsfile.split('.')[0]+"_error.fit", error_image, hdr, overwrite=True)
    
    return error_image


# In[ ]:


def starExtractor(fitsfile, nsigma_value, fwhm_value):
    """
    Extracts stars' x, y positions.
    """
    # check if the region file exists yet, so it doesn't get overwritten
    regionfile = fitsfile.split(".")[0] + ".reg"  
    if os.path.exists(regionfile) == True:
        print(regionfile, "already exists in this directory. Rename or remove the .reg file and run again.")
        return
    
    # Read in the data from the fits file 
    image = fits.getdata(fitsfile)
    # Measure the median absolute standard deviation of the image
    bkg_sigma = mad_std(image)
    # Define the parameters for DAOStarFinder
    daofind = DAOStarFinder(fwhm=fwhm_value, threshold=nsigma_value*bkg_sigma)
    # Apply DAOStarFinder to the image
    sources = daofind(image)
    nstars = len(sources)
    print("Number of stars found in ",fitsfile,":", nstars)
    
    # Define arrays of x-position and y-position
    xpos = np.array(sources['xcentroid'])
    ypos = np.array(sources['ycentroid'])
    
    # Write the positions to a .reg file based on the input file name
    if os.path.exists(regionfile) == False:
        f = open(regionfile, 'w') 
        for i in range(0,len(xpos)):
            f.write('circle '+str(xpos[i])+' '+str(ypos[i])+' '+str(fwhm_value)+'\n')
        f.close()
        print("Wrote ", regionfile)
        return xpos, ypos # Return the x and y positions of each star as variables


# In[ ]:


def measurePhotometry(fitsfile, starxy_pos_list, aperture_radius, sky_inner, sky_outer, error_array):
    """
    Returns a table of photometry values for a list of stars including errors calculated.
    """
    # Read in the data from the fits file
    image = fits.getdata(fitsfile)
    starapertures = CircularAperture(starxy_pos_list,r = aperture_radius)
    skyannuli = CircularAnnulus(starxy_pos_list, r_in = sky_inner, r_out = sky_outer)
    phot_apers = [starapertures, skyannuli]
    
    # including the error array when using aperture_photometry
    phot_table = aperture_photometry(image, phot_apers, error=error_array)
        
    # Calculate mean background in annulus and subtract from aperture flux
    bkg_mean = phot_table['aperture_sum_1'] / skyannuli.area
    bkg_starap_sum = bkg_mean * starapertures.area
    final_sum = phot_table['aperture_sum_0']-bkg_starap_sum
    phot_table['bg_subtracted_star_counts'] = final_sum
    
    # dividing the total error of background by annulus to calculate mean background error in annulus.
    # multiply mean background error in annulus by the area of the star apertures to get total error in aperture.
    bkg_mean_err = phot_table['aperture_sum_err_1'] / skyannuli.area
    bkg_sum_err = bkg_mean_err * starapertures.area
    
    # Creating a new column that represents the error created by the whole background subtraction process. 
    #Using errors in quadrature
    phot_table['bg_sub_star_cts_err'] = np.sqrt((phot_table['aperture_sum_err_0']**2)+(bkg_sum_err**2)) 
    
    return phot_table


# In[ ]:


def ast337_aperture_photometry (all_data_path,       #absolute path to all frames
                                chartname,           # name of this object, used to store all photometry data in an CSV file
                             source_sigma,          #number of sigmas that requires to identify a source
                             frame_parameters,    #list of images that wish to be measure, with their associate parameters in the form of 
                                              #[[name1, fwhm2, apertureradius2, sky ineer radius2, sky outer radius2, filter1],
                                               #[name2, fwhm2, apertureradius2, sky ineer radius2, sky outer radius2, filter2]]
                        ):
    '''
    perform aperature photometry on a set of images of the same patch of the sky, (with different filters)
    '''
    #go to the correct directory
    os.chdir(all_data_path)
    
    #find the coordinates of the sources using the first frame
    xpos, ypos = starExtractor(frame_parameters[0][0], source_sigma,frame_parameters[0][1])
    
    # create empty dataframe to store aperture data
    all_photometry = pd.DataFrame()

    # perform aperature photometry on each image
    for frame in frame_parameters:
        #get the background error
        bgerror=bg_error_estimate(frame[0])
        #measure photometry with position of the first image
        phottable = measurePhotometry(frame[0], np.c_[xpos, ypos], frame[2],frame[3],frame[4],bgerror)
        #add data to data frame
        if frame == frame_parameters[0]:
            all_photometry['id'] = phottable['id']
            all_photometry['xcenter'] = phottable['xcenter']
            all_photometry['ycenter'] = phottable['ycenter']
        #create name of flux columns
        fluxname = '{}flux'.format(frame[5])
        fluxerrname = '{}fluxerr'.format(frame[5])
        all_photometry[fluxname] = phottable['bg_subtracted_star_counts']
        all_photometry[fluxerrname] = phottable['bg_sub_star_cts_err']
        #create column for 1 sec flux
        header = fits.getheader(frame[0])
        flux1name =  '{}flux1s'.format(frame[5])
        all_photometry[flux1name] = all_photometry[fluxname]/header['EXPTIME']
        #create column for error in 1 sec flux
        flux1errname = '{}flux1s_err'.format(frame[5])
        all_photometry[flux1errname] = all_photometry[fluxerrname]/header['EXPTIME']
        #create column for instrumental magnitude
        instmagname = '{}_inst'.format(frame[5])
        all_photometry[instmagname] = -2.5 * np.log10(all_photometry[flux1name])
        #create column for errors in instrumental magnitude
        instmagerrname = '{}_inst_err'.format(frame[5])
        all_photometry[instmagerrname] = 2.5 * 0.434 * (all_photometry[flux1errname]/ all_photometry[flux1name])

    # export everything to a csv file for later use
    all_photometry.to_csv(chartname)
        
    return all_photometry


# In[ ]:


def zero_calculation (zp_parameter,   #list of parameters of zero points in differnt filters, in the form of 
                                      #[[filter1, zeropoint magnitude1, zeropoint err2],
                                      # [filter2, zeropoint magnitude2, zeropoint err2]]
                      dataframe,      # the data frame of the sky that we want to measure, not the standard star
                      chartname):     #name of the chart for saving
    '''
    calculate the "true" magnitude of stars with instrumental zeropoint magnitude given
    '''
    true_mag = pd.DataFrame()
    for fil in zp_parameter:
        #find the instrumental magnitude in certain filters
        instname = '{}_inst'.format(fil[0])
        insterrname = '{}_inst_err'.format(fil[0])
        #calculate the true magnitude from the zeropoints given
        mag = dataframe[instname] + fil[1]
        mag_err = np.sqrt(((dataframe[insterrname]))**2 + (fil[2])**2)    
        #save the value to the data frame
        magname = '{}mag'.format(fil[0])
        magerrname = '{}mag_err'.format(fil[0])
        true_mag[magname] = mag
        true_mag[magerrname] = mag_err
    
    # save these values to a seperate csv file   
    true_mag.to_csv(chartname)
    
    return true_mag

