#Calibrating frames
#THIS CODE WAS NOT WRITTEN BY JACOB OLENICK. IT WAS WRITTEN BY A PREVIOUS STUDENT SUPERVISED BY FRASER CLARKE.

import generalfunctions as gen

import numpy as np
import os
import astropy.units as u
import ccdproc
from ccdproc import CCDData
from astropy.stats import sigma_clipped_stats

#Turn off astropy's annoying warnings that gets printed to the console
import warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)




def combinebias(cal_dir="../Data/20181207/cals", mast_dir=".", filt=[]):
    #Generates master bias file from calibration directory

    #Generate image list
    imagelist = gen.getimages(cal_dir, filt=filt)
    bias_list = []
    for img in imagelist:
        image = CCDData.read(cal_dir + '/' +  img, unit=u.adu)
        if  image.header["IMAGETYP"].strip() == "Bias Frame":
            bias_list.append(image)
            print img
    if len(bias_list) == 0:
	print "ERROR: no bias files"
	return False

    #Create master file	
    master_bias = ccdproc.combine(bias_list, method='average', dtype="float32")
    if not os.path.exists(mast_dir + "/master"):
	os.makedirs(mast_dir + "/master")
    master_bias.write(mast_dir + "/master/master_bias.FIT", overwrite=True)
    '''
    print "Created master_bias file"
    '''
    return True




def combinedark(cal_dir="../Data/20181207/cals", mast_dir=".", filt=[]):
    #Generates master dark file from calibration directory - uses max exptime dark images

    #Get master bias file
    if os.path.isfile(mast_dir + "/master/master_bias.FIT") != True:
        print "No master bias file"
        return False
    master_bias = CCDData.read(mast_dir + "/master/master_bias.FIT")
    
    #Generate image list
    imagelist = gen.getimages(cal_dir, filt=filt)
    full_dark_list = []
    for img in imagelist:
        ccd = CCDData.read(cal_dir + '/' + img, unit=u.adu)
        if  ccd.header["IMAGETYP"].strip() == "Dark Frame":
            img_exptime = ccd.header["EXPTIME"]
            ccd = ccdproc.subtract_bias(ccd, master_bias)
            full_dark_list.append([ccd, img_exptime])
            print img
        
    #Select only those with max exptime
    exptime = max(full_dark_list, key=lambda x: x[1])[1]
    dark_list = [x[0] for x in full_dark_list if x[1] == exptime]
    if len(dark_list) == 0:
	print "ERROR: no dark files"
	return False

    #Generate master file
    master_dark = ccdproc.combine(dark_list, method='median', dtype="float32")
    master_dark.write(mast_dir + "/master/master_dark.FIT", overwrite=True)
    '''
    print "Created master_dark file with " + str(exptime) + " exptime"
    '''
    return True




def combineflat(cal_dir="../Data/20181207/cals", mast_dir=".", filt=[], binning=2):
    #Generates master flat file from calibration directory

    #Get master bias and dark files
    if os.path.isfile(mast_dir + "/master/master_bias.FIT") != True:
        print "No master bias file"
        return False
    if os.path.isfile(mast_dir + "/master/master_dark.FIT") != True:
        print "No master dark file"
        return False
    master_bias = CCDData.read(mast_dir + "/master/master_bias.FIT")
    master_dark = CCDData.read(mast_dir + "/master/master_dark.FIT")

    #Generate image list
    imagelist = gen.getimages(cal_dir, filt=filt)
    flat_list = []
    for img in imagelist:
        ccd = CCDData.read(cal_dir + '/' + img, unit=u.adu)
        if ccd.header["IMAGETYP"].strip() == "FLAT":
            #Rebin images if needed
            if ccd.header["XBINNING"] > binning:
                print "ERROR: Binning too low"
                return False
            elif ccd.header["XBINNING"] < binning:
                ccd.data = gen.rebin(ccd.data, oldbin=ccd.header["XBINNING"], newbin=binning)
                ccd.header["XBINNING"] = binning
                ccd.header["YBINNING"] = binning

            #Remove bias and dark effects
            ccd = ccdproc.subtract_bias(ccd, master_bias)
            ccd = ccdproc.subtract_dark(ccd, master_dark, \
                                        dark_exposure=master_dark.header["EXPTIME"]*u.s, \
                                        data_exposure=ccd.header["EXPTIME"]*u.s, scale=True)
            ccd.data = ccd.data/np.median(ccd.data)
            flat_list.append(ccd)
            
    if len(flat_list) == 0:
	print "ERROR: no flat files"
	return False

    #Generate master file
    master_flat = ccdproc.combine(flat_list, method='median', dtype="float32")
    master_flat.write(mast_dir + "/master/master_flat_" + ".FIT", \
                      overwrite=True)
    '''
    print "Created master_flat file for " + filter_name + " filter"
    '''
    return True




def reduceframes(img_dir="../Data/20181207/imgs", mast_dir=".", mast_cal_dir=False):
    #Removes effects from bias, dark, and flat master files and makes calibrated images
    
    #Get master bias and dark frames - get flat later once have filter_name
    if not mast_cal_dir:
        mast_cal_dir = mast_dir
    if os.path.isfile(mast_cal_dir + "/master/master_bias.FIT") != True:
        print "No master bias file"
        return False
    if os.path.isfile(mast_cal_dir + "/master/master_dark.FIT") != True:
        print "No master dark file"
        return False
    master_bias = CCDData.read(mast_cal_dir + "/master/master_bias.FIT")
    master_dark = CCDData.read(mast_cal_dir + "/master/master_dark.FIT")
#    if not os.path.exists(mast_dir + "/frames"):
#        makedirs(mast_dir + "/frames")
    
    #Reduce images
    raw_image_names = gen.getimages(img_dir, filt=[])
    for img in raw_image_names:
        print img
        ccd = CCDData.read(img_dir + '/' + img, unit=u.adu)
        
        #Mask saturated pixels - not doing any more
        '''mask_saturated = (ccd.data > 50000)
        .data = np.array(ccd.data, dtype=np.float32)
        .data[mask_saturated] = np.nan'''
        
        ccd = ccdproc.subtract_bias(ccd, master_bias)
        ccd = ccdproc.subtract_dark(ccd, master_dark, \
                                        dark_exposure=master_dark.header["EXPTIME"]*u.s, \
                                        data_exposure=ccd.header["EXPTIME"]*u.s, scale=True)
        mean, background, std = sigma_clipped_stats(ccd.data, sigma=3.0, iters=5)
        ccd.data = ccd.data - background
        ccd.data = ccd.data/ccd.header["EXPTIME"]
        ccd.unit = u.adu/u.s
        
        #Add info about background and raw image name to header
        ccd.header['SKY'] = background
        ccd.header['RAWFILE'] = img
    	
    	#Save calibrated frame
        ccd.write(mast_dir + '/frames/' + img[:-4] + '-calibrated.FIT' , overwrite=True)

    print "Created all calibrated frames in " + mast_dir + '/frames'

    return True




def reduceobservation(img_dir, mast_dir, cal_dir=False, \
                      flatfilters=[['V', ['Flat', 'B2']], ['Grating', ['Flat', 'B1']]]):
    #Groups combine functions into one and reduces all frames for an observation

    #Get calibration directory if none given
    if not cal_dir:
        cal_dir = img_dir.split('/')[:-1]
        cal_dir.append('Calibration')
        cal_dir = '/'.join(cal_dir)

    #Reduce frames
    combinebias(cal_dir=cal_dir, mast_dir=mast_dir)
    combinedark(cal_dir=cal_dir, mast_dir=mast_dir)
    for f in flatfilters:
        combineflat(cal_dir=cal_dir, mast_dir=mast_dir, filter_name=f[0], filt=f[1])
    reduceframes(img_dir=img_dir, mast_dir=mast_dir)
    return True
