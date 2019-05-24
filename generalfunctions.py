#General functions
#THIS CODE WAS NOT WRITTEN BY JACOB OLENICK. IT WAS WRITTEN BY A PREVIOUS STUDENT SUPERVISED BY FRASER CLARKE.

import numpy as np
import os
from astropy.io import fits




def getimages(directory, filt=[]):
    #Gets list of image filenames in directory
    
    imagelist = [x for x in os.listdir(directory) if all(f in x for f in filt)]
    return sorted(imagelist)




def makeimage(img, filename):
    #Makes a new fits image from np array
    
    hdu = fits.PrimaryHDU(img)
    if filename[-5:] == '.fits':
        filename = filename[:-5]
    if filename[-4:] != '.fts':
        filename += '.fts'
    hdu.writeto(filename)
    return True




def getarray(filename):
    #Gets array form of image from filename - just easier name
    
    return fits.open(filename)[0].data




def rebin(img, oldbin=1, newbin=2):
    #Changes binning of fits image

    #Find binning ratio
    ratio = newbin/float(oldbin)
    if ratio % 1 != 0:
        print "ERROR: New bin size not multiple of old"
        return False
    ratio = int(ratio)
    
    #Resize for new binning by getting rid of extra columns/rows
    yedge = img.shape[0] % ratio
    xedge = img.shape[1] % ratio
    img = img[yedge:, xedge:]

    #Reshape array and average into new binning
    binimg = np.reshape(img, (img.shape[0]/ratio, ratio, img.shape[1]/ratio, ratio))
    binimg = np.mean(binimg, axis=3)
    binimg = np.mean(binimg, axis=1)
    return binimg
