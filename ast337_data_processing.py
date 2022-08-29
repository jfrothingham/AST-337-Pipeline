#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Modulus for data reduction, image combination, and aperature photometery
#Alyssa Guzman, Molly Loughney, Julia Frothingham, Jingyi Zhang
#2021-11-26
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import astropy
from astropy.io import fits 
from astropy.visualization import ZScaleInterval
import matplotlib.cm as cm
import glob 
import os
import pandas as pd
import scipy.ndimage.interpolation as interp
from matplotlib import colors

from astropy.stats import mad_std
from astropy.stats import sigma_clip
from photutils.utils import calc_total_error
import astropy.stats as stat
from photutils import aperture_photometry, CircularAperture, CircularAnnulus, DAOStarFinder


# In[2]:


def mediancombine(filelist):
    '''
    Creates a median file from multiple calibration images. 
    Used for creating master calibration image and create stacked data image
    '''
    # get the number of the files in the list
    n = len(filelist)
    # Stores the data of the first image to the variable first_frame_data 
    first_frame_data = fits.getdata(filelist[0])
    # Get the dimension of the image from the hearder
    imsize_y, imsize_x = first_frame_data.shape
    # Create a empty array of desired size to store all images
    fits_stack = np.zeros((imsize_y, imsize_x , n)) 
    # Insert the data from each image to the empty stack
    for ii in range(0, n):
        im = fits.getdata(filelist[ii])
        fits_stack[:,:,ii] = im
    # Get the median of each pixel through the stack, collapse into a master file.      
    med_frame = np.nanmedian(fits_stack, axis = 2)
    
    return med_frame


# In[3]:


def bias_subtract(filename, path_to_bias):
    '''
    Loads image to be processed and finds the master bias file based on the inputted path.
    Subtracts bias file from raw image and saves processed image with a new filename.
    '''
    # Load unprocessed image
    frame_data = fits.getdata(filename)
    frame_header = fits.getheader(filename)
    # Load master bias frame
    bias_data = fits.getdata(path_to_bias)
    # Bias subtract
    new_file = frame_data - bias_data
    # Write to new fits file, "b" indicating the bias subtracted version
    new_file_name = 'b_'+filename
    fits.writeto(new_file_name, new_file, frame_header, overwrite=True)  

    return 


# In[4]:


def dark_subtract(filename, path_to_dark):
    '''
    Loads image to be processed and finds the master dark file based on the inputted path.
    Subtracts dark file from raw image and saves processed image with a new filename.
    '''
    # Load raw image
    frame_data = fits.getdata(filename)
    frame_header = fits.getheader(filename)
    # Load master dark frame
    dark_data = fits.getdata(path_to_dark)
    # Dark subtraction
    new_file = frame_data - dark_data
    #write to new fits file, "d" indicating the dark subtracted version
    new_file_name = 'd'+filename
    fits.writeto(new_file_name, new_file, frame_header, overwrite=True)  
    
    return 


# In[5]:


def flat_divide(filename, path_to_flat):
    '''
    Loads image to be processed and finds the master dark file based on the inputted path.
    divide flat field from raw image and saves processed image with a new filename.
    '''
    # Load raw image
    frame_data = fits.getdata(filename)
    frame_header = fits.getheader(filename)
    # Load master flat field
    flat_data = fits.getdata(path_to_flat)
    # Flat division
    new_file = frame_data / flat_data
    # Write to new file, "f" indicating the flat divided image
    new_file_name = 'f'+filename
    fits.writeto(new_file_name, new_file, frame_header, overwrite=True)  
    
    return 


# In[6]:


def norm_combine_flats(filelist):
    '''
    Combine multiple flat field image and normalized the whole field. Create Master flat image.
    '''
    # finds the number of flats
    n = len(filelist)
    # Gets the header of one of the files
    first_frame_data = fits.getdata(filelist[0])
    # Figures out the size of the image
    imsize_y, imsize_x = first_frame_data.shape
    # Creates a new array of the same size with the number of levels (z axis) corresponding to the number of files.
    fits_stack = np.zeros((imsize_y, imsize_x , n)) 
    # for each image, normalizes image by dividing by median value and adding it to the new array
    for ii in range(0, n):
        im = fits.getdata(filelist[ii])
        norm_im =  im/np.median(im)
        fits_stack[:,:,ii] = norm_im   
    # Combines by finding the median along the z axis, collapsing the datacube   
    med_frame = np.median(fits_stack, axis=2)
    
    return med_frame


# In[7]:


def make_master_bias (biases):   # Input naming format of all raw bias image
    '''
    Input all raw bias frames.
    Create master bias frame.
    '''
    # Find all bias frames
    bias = glob.glob(biases)
    bias.sort()
    # Create median combine file
    mbias = mediancombine(bias)
    
    # Use the header from the first raw bias frame; create master bias field
    mbiasheader = fits.getheader(bias[0])
    master_bias = 'MasterBias.fit'
    fits.writeto(master_bias, mbias, mbiasheader, overwrite=True)  
    return mbias


# In[8]:


def make_master_dark (darks):  # Input naming format of all raw darks
    '''
    Input all raw dark images
    Create master dark image with 1 second exposure
    '''
    # Find all raw darks
    dark = glob.glob(darks)
    dark.sort()
    # Bias subtract dark frames, thus the processed image only contain noise from dark current. 
    for frame in dark:
        bias_subtract(frame,'MasterBias.fit')
    # Find all bias subtracted darks    
    bdark = 'b*'+darks
    cal_dark = glob.glob(bdark) 
    
    # Create a list to hold all dark image that has been normalized to 1 second exposure
    dark1s = []    
    #
    for frame in cal_dark :
        # get one frame
        data = fits.getdata(frame)
        header = fits.getheader(frame)
        # find the exposure time of this frame
        exp = header['EXPTIME']
        # Divid each pixel value with the image's exposure time. This gives out an image equivalent to one second exposure
        data1s = data/exp
        # Change the header, set the exposure time to one second
        header['EXPTIME'] = 1
        header['EXPOSURE'] = 1
        # Write to a new fit file, "1s" indicating this is a one secodn exposure
        name = '1s'+frame
        fits.writeto(name, data1s, header, overwrite=True)  
        # Add the name to the list of all 1 second dark
        dark1s.append(name)
    

    # median combine all the 1 second exposure image
    mdark = mediancombine(dark1s)
    # Create master dark file, the master file has 1 second exposure. 
    mdarkheader = fits.getheader(dark1s[0])
    master_dark = 'MasterDark.fit'
    fits.writeto(master_dark, mdark, mdarkheader, overwrite=True) 
    
    # delete all intermidiate images to free up space in computer
    for inter in glob.glob('*b*'+darks):
        os.remove(inter)#delete intermediate 
        
    return mdark


# In[9]:


def make_master_flat (flats, filters):
    '''
    Input all raw flat images and filter names
    Create normalized master flat image for each filter
    '''
    flat = glob.glob(flats)
    flat.sort()
    
    #select flats of different filters
    for fil in filters :
        #create empty array for flats in specofoc filters
        flat_in_fil = []
        # check the header of inidivual frame, if it is taken with the desired filter, add it to the list above
        for frame in flat:
            header = fits.getheader(frame)
            filter_of_this = header['FILTER']
            if filter_of_this == fil:
                flat_in_fil.append(frame)
        
        
        #bias subtract        
        for frames in flat_in_fil:
            bias_subtract(frames,'MasterBias.fit')
        #list of all bias subtracted flat in this filter
        bflatname = 'b*'+flats
        b_flats = glob.glob(bflatname)
        
        # scale flats to 1 second so that we can use the 1 secodn master dark
        flat1s = []
        for frame in b_flats:
            # get the exposure time of each image from their header. 
            header = fits.getheader(b_flats[0])
            time = header['EXPOSURE']
            time = int(time)
            ori_flat = fits.getdata(frame)
            #divide each pixel value by the exposure time, so we get a "1 second exposure" flat 
            scale_flat = ori_flat/time
            #create new header accordinglt
            scale_flat_name = '1s'+frame
            header['EXPOSURE'] = 1
            header['EXPTIME'] = 1
            #create new 1 sec flats
            fits.writeto(scale_flat_name, scale_flat, header, overwrite=True)
            flat1s.append(scale_flat_name)
        
        #dark subtract
        for frames in flat1s:
            dark_subtract(frames,'MasterDark.fit')
        dbflatname = 'd*'+flats
        db_flats = glob.glob(dbflatname)
        
        #median combine, normalized, then create master flat for this filter
        mflat = norm_combine_flats(db_flats)
        mheader = fits.getheader(db_flats[0])
        master_flat = 'Master{}flat.fit'.format(fil)
        fits.writeto(master_flat, mflat, mheader, overwrite=True)  
        
        # delete all intermediate images 
        for inter in glob.glob('*b*'+flats):
            os.remove(inter)


# In[10]:


def data_reduction (all_data_path, #path to directory with all images, including data and calibrations (str)
                    biases,        #naming format of bias frames (str)
                    darks,         #naming format of dark frames (str)
                    flats,         #naming format of flat fields (str)
                    datas,         #naming format of the data that wished to be processed (str)
                    filters):      #name of filters, should match the format in fits header (list)
    '''
    function combining all steps of datat reduction
    input all calibration / data image 
    create master calibration images ready to use
    and fully reduced data images
    '''
    #go to the diretory that contains everything
    os.chdir(all_data_path)
    #create master calibration images
    make_master_bias (biases)
    make_master_dark (darks)
    make_master_flat (flats, filters)
    # get data images 
    data = glob.glob(datas)
    data.sort()
    
    #seperate data frames by their filters.
    for fil in filters :
        #create empty list to hold image with proper filter
        data_in_fil = []
        #add image with proper filter to abovre list
        for frame in data:
            header = fits.getheader(frame)
            filter_of_this = header['FILTER']
            if filter_of_this == fil:
                data_in_fil.append(frame)
                
        #bias subtract        
        for frame in data_in_fil:
            bias_subtract(frame,'MasterBias.fit')
        bdataname = 'b*'+datas
        b_data = glob.glob(bdataname)
        
        
        #create new master dark with exposure time matching the data frame in this filter
        header = fits.getheader(data_in_fil[0])
        time = header['EXPOSURE']
        time = int(time)
        dark = fits.getdata('MasterDark.fit')
        darkheader = fits.getheader('MasterDark.fit')
        #change the exposure time in fits heade accordingly
        darkheader['EXPTIME']=time
        darkheader['EXPOSURE']=time
        #multiply the pixel values by the exposure time to get scaled dark
        scale_dark = dark * time
        scale_dark_name = 'MasterDark{}.fit'.format(time)
        #create new master darks with different exposuretime 
        fits.writeto(scale_dark_name, scale_dark, darkheader, overwrite=True)
        
        #dark subtract with new scaled darks 
        darkname = 'MasterDark{}.fit'.format(time)
        for frames in b_data:
            dark_subtract(frames,darkname)
        dbdataname  = 'd*'+datas    
        db_data = glob.glob('d*')
        
        #flat division with proper filter
        flatname = 'Master{}flat.fit'.format(fil)
        for frames in db_data:
            flat_divide(frames,flatname)
            
        fdbdataname = 'f*'+datas
        
        #remove all intermediate step images
        for inter in glob.glob('d*'+datas):
            os.remove(inter)
        for inter in glob.glob('b*'+datas):
            os.remove(inter)


# In[11]:


def cal_centroid(image,       # numpy array data of an image 
                 sigxy,       # estimate coordinate of reference star in the format of [star1x,star1y]
                 backxy,      # estimate coordinate of a path of background, same format as sigxy
                 boxsize = 100): #size of subframe that contain the refernce star, default to be 100 * 100 pixels
    '''
    Take in an image, a approximate coordinate of the reference star, 
    and a approximate coordinates of a patch of back ground.
    return the coordinates of the centriod of the signal
    '''
    halfsize = boxsize//2
    #take a  sub frame (bozsize) for both signal and background. 
    #note that for signal frame, one more pixel is taken into the subframe to check for continuity of the signal
    sig_sub = image[sigxy[0]-halfsize-1:sigxy[0]+halfsize+1,sigxy[1]-halfsize-1:sigxy[1]+halfsize+1]
    back_sub = image[backxy[0]-halfsize:backxy[0]+halfsize,backxy[1]-halfsize:backxy[1]+halfsize]
    
    #compute the median and standard deviation of the back ground
    #only pixels with three sigma higher than the median of the background is considered to be signal
    back_mean = np.median(back_sub)
    back_std = np.std(back_sub)
    llimit = back_mean + 3 * back_std
    
    #create empty arrays to hold signal values
    sig_sum = np.array([])
    sigx_sum = np.array([])
    sigy_sum = np.array([])
    for i in range(1,101):
        for j in range(1,101):
            if sig_sub[i][j] > llimit:
                #check if the pixel is merely a "hot point" or actual signal
                #If the pixel is above 3 signal but do not have any neighbouring qualified pixels, it does not count as signal
                if sig_sub[i-1][j] > llimit or sig_sub[i+1][j] > llimit or sig_sub[i][j+1] > llimit or sig_sub[i][j+1] > llimit:
                    #caculate the centroid 
                    sig = sig_sub[i][j] - back_mean
                    sigx = sig*j
                    sigy = sig*i
                    #add signal values to corresponding arrays
                    sig_sum = np.append(sig,sig_sum)
                    sigx_sum = np.append(sigx,sigx_sum)
                    sigy_sum = np.append(sigy,sigy_sum)
                #set all other pixels to 0 so that we can check the actual star in the image below
                else:
                    sig_sub[i][j] = 0
            else:
                sig_sub[i][j] = 0
                

    #calculate the coordinates of the centroid, and transform the coordinates back to the original image
    x_cen = np.sum(sigx_sum)/np.sum(sig_sum)
    y_cen = np.sum(sigy_sum)/np.sum(sig_sum)
    x = backxy[0] - halfsize-1 + x_cen
    y = backxy[1] - halfsize-1 + y_cen
    
    #show the background and the cleared signal
    '''
    f, img = plt.subplots(1,2)
    img[0].imshow(back_sub,cmap = 'jet')
    img[1].imshow(sig_sub, cmap = 'jet')
    img[1].plot(x_cen,y_cen,'r*')
    '''
    return(x,y)


# In[12]:


def cal_shift(reference, data, standard_stars):
    '''
    takes in:
    -a reference frame
    -a series of frames we would like to stack on
    -a array of [star coordinates, background coordinates]  in the format [[[ref1x,ref1y],[bg1x,bg1y]],[[ref2x,ref2y],[bg2x,bg2y]]]
    
    returns:
    -shifts in x/y coordinates for each data frame
    '''
    #create an empty arrays to hold the centroids of each standard stars in the reference frame.
    cenx_ref = np.array([])
    ceny_ref = np.array([])
    #calculate the centroid in refernce frame and add to the cen_ref
    for star in standard_stars:
        cen = cal_centroid(reference,star[0],star[1])
        cenx_ref = np.append(cenx_ref, cen[0])
        ceny_ref = np.append(ceny_ref, cen[1])
        
       
    #create empty lists to hold the centroids of each standard stars in each image we want to shift
    cenx_data = []
    ceny_data = []
    #calculate the centroids
    #add the coordinates to the list, with each element(a 1d array) as  different image
    for im in data:
        centx = np.array([])
        centy = np.array([])
        for star in standard_stars:
            cen = cal_centroid(im,star[0],star[1])
            centx= np.append(centx, cen[0])
            centy= np.append(centy, cen[1])
        cenx_data.append(centx)
        ceny_data.append(centy)
    #stack the list into a 2 d array
    cenx = np.stack(cenx_data)
    ceny = np.stack(ceny_data)

    
    #create empty list to hold the shifts calculated from each standard star
    Xshift = []
    Yshift = []
    #calculate the shifts with respect to each standard stars
    #add the shifts to the list, with each element(a 1 d array) represent a different image
    for cen in cenx:
        shifts = np.array([])
        for single in range(0, len(cen)):
            shift = cenx_ref[single] - cen[single]
            shifts = np.append(shifts,shift)
        Xshift.append(shifts)
    for cen in ceny:
        shifts = np.array([])
        for single in range(0, len(cen)):
            shift = ceny_ref[single] - cen[single]
            shifts = np.append(shifts,shift)
        Yshift.append(shifts)
    #stack the list to a 2d array
    xshift = np.stack(Xshift)
    yshift = np.stack(Yshift)
        
       
    #take the mean of all the shifts calculated from each standard star. 
    #This would be the "master shift" that appls to the data frame.
    list_shift = []
    for single in range(0,len(xshift)):
        mshift = np.array([np.mean(xshift[single]),np.mean(yshift[single])])
        r_mshift = np.array([np.mean(yshift[single]),np.mean(xshift[single])])
        list_shift.append(r_mshift)
    master_shift = np.stack(list_shift)
    
    #returns the shifts of pixel values needed to be apply on each data frames 
    
    return master_shift


# In[13]:


def shift (frame,  #name of the image that needed to be shifted 
           shifts, #shifts in x and y direction
           pad):   #pading size suitable to the raw image so all data can be preserved after shifting
    '''
    Create new shifted image
    '''
    # get the unshifted image
    img = fits.getdata(frame)
    # pad the image with a "frame" so no data would be lost after shifting
    new_img = np.pad(img, pad, 'constant', constant_values = -0.001)
    # Shift the image through interpolation, change the image with no data to NaN
    shift_img = interp.shift(new_img, (shifts[0], shifts[1]), cval = -0.001)
    #shift_img[shift_img <= 0.00001] = np.nan
    # modify the image size in the fits header
    header = fits.getheader(frame)
    header['NAXIS1'] = header['NAXIS1'] = 2 * pad
    header['NAXIS2'] = header['NAXIS2'] = 2 * pad
    # create new fits file for shifted image
    shift_name = 's_'+frame
    fits.writeto(shift_name, shift_img, header, overwrite=True)   
    return 


# In[14]:


def stack (name,              # create a name for the stacked image
           frames,            # all frames that needs to be stacked 
           norm_ref_stars,    # lists of refernce star and back ground pairs, in the format [[[ref1x,ref1y],[bg1x,bg1y]],[[ref2x,ref2y],[bg2x,bg2y]]]
           pad):              # padding size to image, default to be 50 pixels on each sides
    '''
    Takes in images and coordinates of refernce stars, 
    shift and stack image together
    '''
    
    #change the axis. As the x axis seen in DS9 is the second axis in np array
    ref_stars = []
    for pairs in norm_ref_stars:
        reverse = [[pairs[0][1],pairs[0][0]],[pairs[1][1],pairs[1][0]]]
        ref_stars.append(reverse)
    
    #get data from their names
    rf_frame = fits.getdata(frames[0])
    data_frames = []
    for d in frames:
        data_frames.append(fits.getdata(d))
        
    #calculate shifts for eah frame with respect to the first image
    all_shifts = cal_shift(rf_frame,data_frames,ref_stars)
    # quick sanity check..... 
    print(all_shifts)
    # shift all images with respect to the first image
    for i in range(0,len(frames)):
        shift(frames[i],all_shifts[i],pad)
    # grab all shifted image
    all_shift_img = glob.glob('s_'+frames[0][:3]+'*')
    # stack the shifted images
    stacked_img = mediancombine(all_shift_img)
    header = fits.getheader(all_shift_img[0])
    stacked_name = 'Stacked{}.fit'.format(name)
    fits.writeto(stacked_name, stacked_img, header, overwrite=True)   
    
    for inter in glob.glob('s_'+frames[0][:3]+'*'):
        os.remove(inter)
    
    return stacked_name


# In[15]:


def image_alignment (all_data_path,     # absolute path to the directory that contains all images
                     name,              # naming format of desired stacked images
                     frames,            # naming format of the images that needs to be stacked
                     norm_ref_stars,    # lists of refernce star and back ground pairs, in the format [[[ref1x,ref1y],[bg1x,bg1y]],[[ref2x,ref2y],[bg2x,bg2y]]]
                     filters,           #list of filters
                     pad = 50,          #padding size to image, default to be 50 pixels on each sides
                    ):
    '''
    shift, combine, then align all reduced images.
    
    '''
    #go to the correct directory
    os.chdir(all_data_path)
    # get all images
    all_image = glob.glob(frames)
    all_image.sort()
    # empty list that later holds stacked image in each filter
    all_stacked_image_name = []
    
    # create seperate stacked image for each filter
    for fil in filters:
        #create empty list to hold image with proper filter
        data_in_fil = []
        #add image with proper filter to abovre list
        for frame in all_image:
            header = fits.getheader(frame)
            filter_of_this = header['FILTER']
            if filter_of_this == fil:
                data_in_fil.append(frame)
        #create staked image for each filter
        fil_name = name.format(fil)
        #return the name of the stacked image and add 
        stacked_name = stack(fil_name, data_in_fil, norm_ref_stars, pad)
        all_stacked_image_name.append(stacked_name)
    #sanity check
    print(all_stacked_image_name)    
    
    # calculate the shifts of stacked images with respect to the first one
    #change the axis. As the x axis seen in DS9 is the second axis in np array
    ref_stars = []
    for pairs in norm_ref_stars:
        reverse = [[pairs[0][1]+pad,pairs[0][0]+pad],[pairs[1][1]+pad,pairs[1][0]+pad]]
        ref_stars.append(reverse)
        
    rf_frame = fits.getdata(all_stacked_image_name[0])
    data_frames = []
    for d in all_stacked_image_name:
        data_frames.append(fits.getdata(d))
    
    # calculate shifts for eah frame with respect to the first image
    all_shifts = cal_shift(rf_frame,data_frames,ref_stars)
    # quick sanity check..... 
    print('Shifts of stacked images are :',all_shifts)
    # shift all images with respect to the first one
    for i in range(0,len(all_stacked_image_name)):
        shift(all_stacked_image_name[i],all_shifts[i],pad)    
    
    return

def Isochrones_fitting (mag_file_name,  # name of the file that contains the aparent magnitude of cluster stars
                        cluster_name,   # name of the cluster
                        iso_file_name,  # name of the model isochrone
                        iso_time,       # age of the model 
                        colorindex,     # a list of two filters that used for calculating the color index,
                                        # eg. ['V', 'R'] then V-R would be on the xaxis of the CMD
                        yaxis,          # the filter that the color index should be plotted against
                        magnitude_shifts# the shift in the yaxis  so that the isochrone matches the actual data
                       ):
    '''
    plot the CMD and fits isochrones to it.
    '''
    #load in data from the cluster and the isochrones file
    mag = pd.read_csv(mag_file_name)
    iso = pd.read_csv(iso_file_name, skiprows = 10, delim_whitespace = True)
    #find data with proper filter in cluster data 
    color = mag['{}mag'.format(colorindex[0])] - mag['{}mag'.format(colorindex[1])]
    xerr = np.sqrt((mag['{}mag_err'.format(colorindex[0])]) **2 + (mag['{}mag_err'.format(colorindex[1])])**2)
    ydata = mag['{}mag'.format(yaxis)]
    yerr = mag['{}mag_err'.format(yaxis)]
    #find data with proper filter in isochrones file 
    iso_color = iso['{}'.format(colorindex[0])] - iso['{}'.format(colorindex[1])]
    iso_ydata = iso['{}'.format(yaxis)]
                
    # plot the cmd and fitted isochrones
    fig = plt.figure(figsize = (15,7))
    fig.suptitle('Color Magnitude Diagtram and Isochrone Fitting', fontsize=20)
    
    #create CMD            
    CMD = plt.subplot(1, 2, 1)
    CMD.errorbar(color, ydata, xerr, yerr, marker = 'o', markersize=3., linestyle='None',elinewidth=1,label = cluster_name)
    CMD.set_ylabel('{} Magnitude'.format(yaxis),fontsize=15)
    CMD.set_xlabel('{}-{}'.format(colorindex[0],colorindex[1]),fontsize=15)
    CMD.set_title('Color Magnitude Diagram',fontsize=15)
    CMD.invert_yaxis()
    CMD.legend(loc='upper left',fontsize = 15)
                
    #create isochrone fitting
    ISO = plt.subplot(1, 2, 2)
    ISO.errorbar(color, ydata, xerr, yerr, marker = 'o', markersize=2., linestyle='None',elinewidth=1, label = cluster_name)
    ISO.plot(iso_color, iso_ydata + magnitude_shifts, 'r-', label=('{}yr Isochrone'.format(iso_time)+' shifted by '+str(magnitude_shifts)+' magnitudes'))
    ISO.set_ylabel('{} Magnitude'.format(yaxis),fontsize=15)
    ISO.set_xlabel('{}-{}'.format(colorindex[0],colorindex[1]),fontsize=15)
    ISO.set_title('Isochrone Fitting',fontsize=15)
    ISO.invert_yaxis()
    ISO.legend(loc='upper left', fontsize = 14)