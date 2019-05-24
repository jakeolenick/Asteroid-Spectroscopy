# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 19:31:29 2019

@author: jakeo
"""

import framecalibration as fc

import numpy as np
from astropy import units as u
from astropy.nddata import CCDData
import ccdproc
import scipy.ndimage.interpolation as transform
from matplotlib import pyplot as plt
from scipy import signal as sig
from scipy import optimize as opt
from scipy.linalg import lstsq
from scipy import ndimage as ndim
from scipy import integrate as intgrl
import astropy.stats as st
import astropy.io.fits as fits
import csv

########Instructions:
#>>> xyl = xy_to_lambda()
#
#>>> h_params = h_fit_params()
#these are used to calibrate the 

######################## GET, PLOT, AND COMPARE SPECTRA #######################

def get_spectrum(img,h_params=[],w_arr=[],\
                lambda_bin=1,vtol=3,v_width=10,ysource=None,offset=-625):
    '''Gets a spectrum, given an image. Only takes one spectrum per image (the
    brighter of two objects, if there are two in the image). img must be a file
    path as a string.
    Output: a spectrum, which is a list of two tuples, the first being
    wavelengths, while the second is observations.
    NOTE: offset=-1965 or so for the HR1544 images and their peers, using the
        old arcs
    offset = -625 for the newer batch with the newer arcs.
    '''
    print "start"
    out = []
    hdu = fits.open(img)
#    header = hdu[0].header
#    xbin = header['XBINNING']
#    ybin = header['YBINNING']
    xbin=1
    ybin=1
    arr = np.asarray(CCDData.read(img,unit="adu"))
    arr = ndim.zoom(arr,(xbin,ybin),order=0)
    #find the source
    if (ysource==None):
        imgt = arr.T
        stats = st.sigma_clipped_stats(imgt)
        mean = stats[0]
        std = stats[2]
        peak_tuple = sig.find_peaks(imgt[0],mean+vtol*std,None,5) #want to find highest initial peak, to use as starting guess
        p = peak_tuple[0]
        h = peak_tuple[1]['peak_heights']
        hmax = 0
        hmloc = -1
        for i in range(0,len(h)):
            if (h[i] > hmax):
                hmax = h[i]
                hmloc = i
        if (hmloc==-1):
            print "Could not find source"
            return out
        else:
            ystart = p[hmloc]
    else:
        ystart = ysource
    #trace along it, from ystart, like you do in xy_to_h, and get spectrum using
    #xy_to_lambda
    print "source at: " + str(ystart)
    if(len(h_params)==0):
        print "No h_fit_params given!"
        m1,m2,m3,b = h_fit_params()
    else:
        m1=h_params[0]
        m2=h_params[1]
        m3=h_params[2]
        b=h_params[3]
    print "parameters found"
    wavflux=[] # will be an array of 2-tuples of wavelengths and fluxes
    #no distinction is made at this point between subsequent columns extracted
    #from the image.
    if (len(w_arr)==0):
        print "No xy_to_lambda mapping given!"
        wav_arr = xy_to_lambda()
    else:
        wav_arr=w_arr
    for i in range(0,len(arr[0])):
        y = ystart + m1*i**3+m2*i**2+m3*i #e.g. 112.43
        ylow = int(y-v_width) #e.g. 107
        lowfrac = ylow - (y-v_width) + 1 #e.g. 107 - 107.43 + 1 = 0.67
        yhigh= int(y+v_width) #e.g. 117
        highfrac = (y+v_width)-yhigh #e.g. 117.43 - 117 = 0.43
        wavflux.append((wav_arr[ylow][i],arr[ylow][i]*lowfrac))
        for j in range(ylow+1,yhigh): #the plus 1 makes it: (v_width*pixels),mid pixel,(v_width*pixels)
            wavflux.append((wav_arr[j][i],arr[j][i]))
        wavflux.append((wav_arr[yhigh][i],arr[yhigh][i]*highfrac))
            
    wavflux.sort()
    new=[]
    Warr=np.asarray(wavflux)
    Warr=Warr.T
    #average it out
    for i in range(0,len(wavflux),(2*v_width+1)*lambda_bin):
        wav = Warr[0][i:i+(2*v_width+1)*lambda_bin].mean() + offset
        flu = abs(Warr[1][i:i+(2*v_width+1)*lambda_bin].sum())
        new.append((wav,flu))
    #instead of 765 tuples of length 2, make it 2 of length 765
    final = zip(*new)
    #resample evenly
    mi = min(final[0])
    ma = max(final[0])
    delwav = (ma-mi)/len(final[0])
    newxs=np.arange(mi,ma,delwav)
    newys=np.interp(newxs,final[0],final[1])
    newxs=tuple(newxs)
    newys=tuple(newys)
    truefinal=[newxs,newys]
    
    return truefinal


def plot_spectrum(z,units="physical",exptime=1,dia=40,gain=2.65,starname=""\
                  ,secondary_plot=False):
    '''if units are physical, then the exptime, dia, and gain parameters are
    used to convert counts on the detector to erg/cm^2/s/A
    Note: If the image has been cleaned up, it is ALREADY divided by exposure
    time.
    Secondary_plot should only be used after a primary plot has been drawn.
    '''
    #z[0] is a tuple of wavelengths, z[1] is one of counts
    if (secondary_plot): #allows you to plot two spectra with different y ranges
       axR = plt.gca().twinx()
    if (units == 1):
        if (secondary_plot):
            axR.plot(*z,color="orange",label=starname)
        else:
            plt.plot(*z,label=starname)
        plt.xlabel("Wavelength (Angstroms)")
        plt.title("Spectrum of the object " +starname+ " at each wavelength")
        plt.legend()
        return z
    if (units == "log"):
        new = [z[0],()]
        for i in range(0,len(z[0])):
            fl = z[1][i]
            lnf = np.log(fl)
            new[1] = new[1]+(lnf,)
        if (secondary_plot):
            axR.plot(*new,color="orange",label=starname)
        else:
            plt.plot(*new,label=starname)
        plt.xlabel("Wavelength, "+"$\AA$")
        plt.ylabel("lnF,\n erg cm^-2 s^-1 "+ '$\AA$' +"^-1")
        plt.title("Observed flux, per wavelength interval\n")
        plt.legend()
        return new       
    elif (units=="ADU"):
        if (secondary_plot):
           axR.plot(*z,color="orange",label=starname)
        else:
            plt.plot(*z,label=starname)
        plt.xlabel("Wavelength, "+"$\AA$")
        plt.ylabel("ADU")
        plt.title("Detection events (in ADU)")
        plt.legend()
        return z
    elif (units=="physical" or units=="log physical"):
        h = 6.626e-34
        c = 2.998e8
        new = [z[0],()]
        A = np.pi * ((dia/2)**2)
        delta = z[0][1]-z[0][0]
#        mx = 0
#        my = 0
#        fwhm = 4
#        varx = fwhm**2 / (8 * np.log(2))
#        vary = varx
#        gaussfrac = intgrl.dblquad(gaussian2d,-0.5,0.5,-np.inf,np.inf,args=(mx,my,varx,vary,1))
#        gaussfrac = gaussfrac[0]
        gaussfrac=1
        for i in range(0,len(z[0])):
            wl = z[0][i] * (10**-10)
            f = c / wl
            ph = z[1][i] * gain
            E = ph * h * f * (10**7)
            fl = E / (A*exptime*delta)
            fl = fl / gaussfrac
            if (units=="log physical"):
                lnf = np.log(fl)
            else:
                lnf = fl
            new[1] = new[1]+(lnf,)
        if (secondary_plot):
            axR.plot(*new,color="orange",label=starname)
        else:
            plt.plot(*new,label=starname)
        plt.xlabel("Wavelength, "+"$\AA$")
        if (units=="log physical"):
            plt.ylabel("lnF,\n erg cm^-2 s^-1 "+ '$\AA$' +"^-1")
        else:
            plt.ylabel("Flux,\n erg cm^-2 s^-1 "+ '$\AA$' +"^-1")
        plt.title("Observed flux, per wavelength interval\n")
        plt.legend()
        return new
    else:
        print "units must be 'physical' or 'ADU' or 'log physical' or 'log ADU' or 'log' or the int 1"
        return z

def set_defaults():
    '''sets preferred default font sizes for pyplot
    '''
    plt.rc('font', size=42)          # controls default text sizes
    plt.rc('axes', titlesize=48)     # fontsize of the axes title
    plt.rc('axes', labelsize=48)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=42)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=42)    # fontsize of the tick labels
    plt.rc('legend', fontsize=42)   # legend fontsize
    plt.rc('figure', titlesize=48)  # fontsize of the figure title, does nothing

def compare_spectra(spec1,spec2,indexes=False,smoothing=9,mode="median"):
    '''inputs: two spectra, in the form of 2-element lists of tuples, where the
    first list is the wavelengths, and the second list is the observations.
    The first spectrum is treated as the one to be evaluated, while the second
    is some basic reference spectrum. The two spectra needn't be of the same
    resolution (the "best_lines" function finds closest matches and outputs
    are at the avearge wavelength between the two spectra. e.g. if one has
    observations at 3301,3305,and 3309, and the reference spectrum has an
    observation at 3307.1, it'll compare the data at 3309 and 3307.1)
    indices: indicates whether you want the comparison to be assigned to a
        the wavelength (in the example above, this is 3308.15), or to be assigned
        to the index of the given spectrum (in the example above, this would be
        the index of the observation at 3307.1).
    smoothing: Uses either a median or moving average smoothing to produce a 
        smoother efficiency spectrum.
    mode: if "median" the smoothing is a moving median not a moving average.
        Otherwise, mode is passed directly to convolve. Options are "full" 
        "valid" or "same". "Valid" avoids edge effects, where (averaging 5 at 
        a time) you might be average 3 zeros off the left edge with 2 actual 
        values, which is what you get with "same". "Full" exceeds the edges
        and is not recommended.
    output: a "spectrum" i.e. list of 2 tuples, where the first is wavelengths
    and the second is the ratio of the values in the first to the second.
    '''
    pixlines,theirlines = best_lines(list(spec1[0]),list(spec2[0]),m1=0,m2=1,b=0,indexes=True)
    ratios = ()
    if (indexes):
        inds = ()
        myindex = 0
        matches = []
        for i in range(0,len(pixlines)):
            matches.append(spec1[0][pixlines[i]])
        for i in range(0,len(spec1[0])):
            w = spec1[0][i]
            try: 
                a = pixlines.index(i)
                myindex=pixlines[a]
            except:
                nothing=0
            theirlineindex = pixlines.index(myindex)
            theirindex = theirlines[theirlineindex]
            ratio = (spec1[1][myindex]/spec2[1][theirindex])
            inds = inds+(i,)
            ratios = ratios + (ratio,)
        smoothratios = list(ratios)
        if (mode=="median"):
            smoothratios = sig.medfilt(smoothratios,smoothing)
        else:
            smoothratios = np.convolve(smoothratios,np.ones((smoothing,))/smoothing,mode)
            if (mode=="valid"):
                inds = inds[smoothing/2:len(inds)-smoothing/2]
        out = [inds,smoothratios]
        return out
    else:
        wavs = ()
        for i in range(0,len(pixlines)):
            myindex = pixlines[i]
            theirindex = theirlines[i]
            ratio = (spec1[1][myindex]/spec2[1][theirindex])
            wav = (spec1[0][myindex] + spec2[0][theirindex])/2
            ratios = ratios + (ratio,)
            wavs = wavs + (wav,)
        smoothratios = list(ratios)
        if (mode=="median"):
            smoothratios = sig.medfilt(smoothratios,smoothing)
        else:
            smoothratios = np.convolve(smoothratios,np.ones((smoothing,))/smoothing,mode)
            if (mode=="valid"):
                wavs = wavs[smoothing/2:len(wavs)-smoothing/2]
        out = [wavs,smoothratios]
        return out
     
        
def get_efficiency_curve(myspec,refspec="./Reference Spectra/Reference Spectrum HR 3454.csv",smoothing=1):
    '''Produces an (approximate) absolute efficiency curve.
    myspec: a PHYSICAL spectrum of a chosen reference object. This is the one
        that you took, to see your device's efficiency.
    refspec: this is important. This should be a csv file, formatted so that
        the first row contains labels, the second row is empty, and all rows
        after that contain the wavelength, the flux (times 10**16), anything
        in the third column, and the wavelength binning. This is because that's
        how ESO data is provided (except you need to copy and paste onto a 
        spreadsheet software to get a csv file from it)
        
    output: a spectrum, with binning equal to YOUR spectrum. A spectrum is a
        list of two tuples, where the first tuple contains wavelengths. In
        this case, the second tuple contains an estimated absolute efficiency
        at each wavelength. This can then be fed into the normalize_spectrum
        function.
    '''
    
    csvfile = open(refspec, 'r')
    csvreader = csv.reader(csvfile,delimiter=",")
    wavs = ()
    fluxes = ()
    aslist=[]
    for row in csvreader:
        aslist.append([])
        for item in row:
            aslist[len(aslist)-1].append(item)
    for row in aslist[2:]:
        wav = float(row[0])
        wavs = wavs+(wav,)
        flux = float(row[1]) / (10**16)
        fluxes = fluxes+(flux,)
    csvfile.close()
    mi=min(myspec[0])
    ma=max(myspec[0])
    myrange=ma-mi
    mybin=myrange/len(myspec[0])
    newwavs = np.arange(mi,ma,mybin)
    newfluxes = np.interp(newwavs,wavs,fluxes)
    theirspec=[newwavs,newfluxes]
    effspec = compare_spectra(myspec,theirspec,smoothing=smoothing)
    return effspec

    
def normalize_spectrum(myspec, effspec):
    """Produces a Normalized spectrum using a given efficiency curve.
    myspec: the spectrum to be normalize (a list of two tuples)
    effspec: the efficiency "spectrum" (another list of two tuples).
    """
    out = [myspec[0],()]
    newvals=()
    for i in range(0,len(myspec[0])):
        wav = myspec[0][i]
        val = myspec[1][i]
        #find the nearest wavelength in the efficiency spectrum
        pixlines, theirlines = best_lines([wav],list(effspec[0]),0,1,0,tol=1,indexes=True)
        ind = theirlines[0]
        ratio = effspec[1][ind]
        newval = val/ratio
        newvals = newvals+(newval,)
    out[1]=newvals
    return out
        
def trim_spectrum(spec,start=237,stop=None):
    '''Simple function to return a trimmed spectrum. Makes it easier to get rid
    of the data past 3800A, which is pure noise, and lasts until about the 240th
    index
    '''
    out = [spec[0][start:stop],spec[1][start:stop]]
    return out


####################### BEST FITS FOR HEIGHT AND WAVELENGTH ###################

def h_fit_params(tol=8,cal_img='../Data/20181207/imgs/HR1544-0001SpecPhot.FIT'):
    cal = np.asarray(CCDData.read(cal_img, unit=1))
    cal = ndim.zoom(cal,(2,2))
    img = cal.T
    stats = st.sigma_clipped_stats(cal)
    mean = stats[0]
    std = stats[2]
    peak_tuple = sig.find_peaks(img[0],mean+tol*std,None,5) #want to find highest initial peak, to use as starting guess
    p = peak_tuple[0]
    h = peak_tuple[1]['peak_heights']
    yguess = 100
    hmax = 0
    hmloc = -1
    for i in range(0,len(h)): #should be O(1)
        if (h[i] > hmax):
            hmax = h[i]
            hmloc = i
    yguess = p[hmloc]
    vpa = peak_vert_arr(cal,yguess,tol) #O(n^2)
    m1,m2,m3,b = map_to_h(vpa)
    return m1,m2,m3,b

#XY to lambda and h functions, which effectively make images, whose values are
#either the wavelength, or the height, of an observation at that pixel.

def xy_to_lambda(min_disp = -9.2,max_disp= -8,linelist_file = "Argon Neon line list.txt",\
               cal_img="../Data/20190307/Arc-006HgArNe.fit",end=510,lintol=0.3,\
               max_iters=10,guess = [-0.00005,-8.61,8987],order=2):
    '''auto-calibrates wavelength based on the Argon and Neon lamps. returns a 
    two-dimensional array, of the same shape as the image, where the value at 
    each pixel is the wavelength at said pixel.
    Optional inputs: 
        min_disp = the minimum dispersion (angstroms per pixel) that you expect
        max_disp = the maximum dispersion that you expect
        linelist_file = a csv-formatted file containing the largest set of
         lines to which you're calibrating (which are set, by default, to all 
         the brightest lines visible from an Argon and a Neon lamp, going from 
         the 4047.0A line on Neon to the 7535.8A line on Hg/Ar. Lamps used 
         were Oriel Instruments spectral calibration lamps)
        cal_img = an image which contains at least some of the lines listed in
         linelist_file.
        end = an upper limit on the pixels that you're calibrating to. This is
         because the way our spectrometer was set up meant that there was no 
         image on the CCD beyond pixel 600 or so. This is only to make the pro-
         cess SLIGHTLY quicker.
        lintol = the tolerance for identifying lines with "best_lines", which is
         defined below.
        max_iters = the maximum number of iterations for each time you're 
         looking for what lines you may have detected (it will normally give up 
         iterating when the standard deviation of the fit isn't changing 
         between iterations).
    '''
    lines = read_linelist(linelist_file)
    print lines
    cal = np.asarray(CCDData.read(cal_img, unit=1))
    arr = []
    peaks=get_pix_peaks(cal,0)
    if (order==2):
        m1=guess[0]
    elif (order==1):
        m1=0
    else:
        print "Order must be 1 or 2 for xy_to_lambda!"
        return []
    m2=guess[1]
    b=guess[2]
    poorfits=0
    toofew = 0
    for r in range(0,len(cal),5):
#        binnedRow=[]
#        for x in range(0,len(cal[r])):
#            pix = (cal[r][x]+cal[r+1][x]+cal[r+2][x]+cal[r+3][x]+cal[r+4][x]) / 5.0
#            binnedRow.append(pix)
        binnedRow = cal[r:r+5].mean(axis=0)
        if (r < end):
            peaks = get_pix_peaks(binnedRow)
            ps,ls=best_lines(peaks,lines,m1,m2,b,tol=lintol)
            m1a,m2a,ba=m1,m2,b
            sigm=0.0
            #print len(ps),
            if(len(ps)>2): #want at least 3 points to get a fit
                for i in range(0,max_iters-1):    
                    ps,ls=best_lines(peaks,lines,m1a,m2a,ba,tol=lintol)
                    m1a,m2a,ba=map_to_lambda(ps,ls,order)
                    ps = np.array(ps)
                    guess = ps*ps*m1a + ps*m2a + ba
                    siga = (ls-guess).std()
                    if (abs(siga-sigm)<0.0001) :
                        break
                    sigm=siga
                    #print "|"+str(i)+" " +str(sigm),
                m1a,m2a,ba = map_to_lambda(ps,ls,order)
                #print len(ps)
                if(m2a<max_disp and m2a>min_disp):
                    m1=m1a
                    m2=m2a
                    b=ba
    #               print 'updated!'
                else:
                    print "poorfit at " + str(r),
                    print "m1 was " + str(m1a),
                    print "m2 was " + str(m2a),
                    print "and b was " + str(ba)
                    poorfits=poorfits+1
                row = []
                for x in range(0,len(cal[r])):
                    l = m1 * x**2 + m2 * x + b
                    row.append(l)
            else:
                toofew = toofew+1
                row = []
                for x in range(0,len(cal[r])):
                    l = m1 * x**2 + m2 * x + b
                    row.append(l)
        else:
            row = []
            for x in range(0,len(cal[r])):
                l = m1 * x**2 + m2 * x + b
                row.append(l)
        arr.append(row)
        print str(m1) +", "+str(m2)+", "+str(b)+"|",
    out = ndim.zoom(arr,(5,1),None,1)
    print "poor fits: " + str(poorfits)
    print "too few peaks: " + str(toofew)
    print "total attempts to map wavelength: " + str(len(cal)/5)
    return out

def xy_to_h(tol=8,cal_img='../Data/20181207/imgs/HR1544-0001SpecPhot.FIT'):
    '''auto-calibrates height based on a spectrum taken of HR1544. returns a 
    two-dimensional array, of the same shape as the image, where the value at 
    each pixel is the wavelength at said pixel.
    Optional paramters:
        tol = the tolerance for detecting where the spectrum is in each row,
         in terms of how many standard deviations brighter it is than the mean.
        cal_img = an image of a single source, e.g. a star, in the spectro-
         meter.
    '''
    m1,m2,m3,b = h_fit_params()
    cal = np.asarray(CCDData.read(cal_img, unit=1))
    cal = ndim.zoom(cal,(2,2))
    out=[]
    for i in range(0,len(cal)):
        #H(x,y) = (y-h(0)) + h(x), so H(x,y) is "what y pixel, at x=0, is this
        #pixel the same height as?" Note that h(0) = b always
        out.append([])
        for j in range(0,len(cal[0])):
            hx = m1*j**3 + m2*j**2+m3*j + b
            Hxy = i - b + hx
            out[i].append(Hxy)
    return out



def pix_to_arcsec(l,mode="spectroscopic",sample_col=380):
    """Gives you the number of arcseconds represented by each pixel. 
        l = a list of 2 ntuples. If mode is "guide" or "guide camera" then the 
            first tuple must contain the path names for file to be used in 
            calibrating (each file must be an image taken on the guide camera).
            The second must contain the known arcsecond separations of the two 
            brightest objects in each image given, in the same order.
            If mode is "spectroscopic" or "spectroscopic camera" then the first
            tuple must contain the path names for the files to be used in 
            calibrating (taken with the spectroscopic camera). The second tuple
            must contain the relative angular positions for the images given,
            in the order given. For spectroscopic mode, all images must be of 
            the same star, at different positions on the slit, so that the 
            angular positions are their positions on the slit relative to each
            other.
        mode = string, indcating whether the images are from the guide camera 
            or the spectroscopic camera.
        sample_col = only used in spectroscopic mode. The column from which
            data is sampled in order to determine the height of the object in
            each image.
            
    Output:
        In spectroscopic mode, the two best fit parameters for a linear mapping
        from vertical pixels to angular distances. The b parameter is not
        strictly useful, since the information is only relative. m1 is the
        number of arcseconds per pixel.
        
        
    """
    if (mode=="spectroscopic" or mode=="spectroscopic camera"):
        imgs = []
        for i in l[0]:
            img = np.asarray(CCDData.read(i,unit=1))
            imgs.append(img)
        pixlocs = ()
        arclocs = l[1]
        #5 * std
        for i in range(0,len(imgs)):
            img = imgs[i]
            mean = np.mean(img)
            std = np.std(img)
            print "mean and std are ",
            print mean,std
            peaks,peakdic = sig.find_peaks(img.T[sample_col],\
                                           height = mean+2*std,\
                                           threshold=(None,10*std))
            if (len(peaks)==1):
                print "1 peak found!"
                yloc = peaks[0]
                pixlocs = pixlocs+(yloc,)
            elif (len(peaks) <= 0):
                print "no peaks found!"
                #currently, just removes the faulty sample
                arclocs = arclocs[0:i] + arclocs[i+1:]
            else:
                print "many peaks found!"
                peakindex = np.argmax(peakdic['peak_heights'])
                yloc = peaks[peakindex]
                pixlocs=pixlocs+(yloc,)
        #we now have an array of pixel locations and one of angular locations
        M = []
        y = []
        for i in range(0,len(pixlocs)):
            pix = pixlocs[i]
            arc = arclocs[i]
            M.append([pix**0,pix**1])
            y.append(arc)
        print M
        params = lstsq(M,y)
        b = params[0][0]
        m1 = params[0][1]
    
    elif (mode=="guide" or mode=="guide camera"):
        imgs = []
        for i in l[0]:
            img = np.asarray(CCDData.read(i,unit=1))
            imgs.append(img)
        #find all relative maxima
        pixdists=[]
        arcdists=l[1]
        for n in range(0,len(imgs)):
            img = imgs[n]
            mean = np.mean(img)
            std = np.std(img)
            relmax=[] #[(x1,y1,height1),(x2,y2,height2)...]
            for r in range(0,len(img)):
                rpeaks,rpeakdic=sig.find_peaks(img[r],height=mean+10*std,\
                                                       width=(1.5,None))
                #peaks on that row
                for c in range(0,len(img[r])):
                    #if you're on a column that's a peak on this row, see if this row is a peak on this column
                    if (c in rpeaks):
                        cpeaks,cpeakdic=sig.find_peaks(img.T[c],height=mean+10*std,\
                                                       width=(1.5,None))
                        if (r in cpeaks):
                            ind = cpeaks.tolist().index(r)
                            height = cpeakdic['peak_heights'][ind]
                            relmax.append((c,r,height))
            #should now have an array of 2 tuples
            if(len(relmax)>=2):
                if(len(relmax)>2):
                    #if more, take two brightest maxima
                    print "too many maxima!"
                    print relmax
                    swapped = zip(*relmax)
                    swapped[2]=list(swapped[2])
                    maxin1 = np.argmax(swapped[2])
                    max1 = relmax[maxin1]
                    swapped[2][maxin1]=-np.inf
                    maxin2 = np.argmax(swapped[2])
                    max2 = relmax[maxin2]
                    relmax = [max1,max2]
                max1 = relmax[0]
                max2 = relmax[1]
                dist = ((max1[0]-max2[0])**2 + (max1[1]-max2[1])**2)**0.5
                pixdists.append(dist)
            elif(len(relmax)<2):
                print "finding maxima failed!"
                arcdists=arcdists[0:i] + arcdists[i+1:]
                
            
            
        M = []
        y = []
        for i in range(0,len(pixdists)):
            pix = pixdists[i]
            arc = arcdists[i]
            M.append([pix**0,pix**1])
            y.append(arc)
        print M
        params = lstsq(M,y)
        b = params[0][0]
        m1 = params[0][1]
    else:
        print "Mode must be either \"guide\" (or \"guide camera\") or \"spectroscopic\" (or \"spectroscopic camera\")"
        m1 = None
        b = None
    
    return m1,b



################ FINDERS OF PEAKS IN A GIVEN ROW OR COLUMN ####################

#functions which produce arrays, which contain the gauss-fitted pixel locations
#of peaks in the image, either horizontally ("get_pix_peaks") or vertically
#("vert_pix_peaks)    

def get_pix_peaks(img,y=None):
    '''gets gauss-fitted peak locations, for a given row on an image, or if you
    give it just a row it finds the peaks there
    '''
    if (y==None and ((type(img[0]) == int) or (type(img[0])==float) or (type(img[0])== np.float64))):
        data = img
    elif(y!=None):
        data = img[y]
    else:
        print "get_pix_peaks failed: can only gets peaks on a given row."
        return None
    stats = st.sigma_clipped_stats(data)
    mean = stats[0]
    std = stats[2]
    peak_tuple = sig.find_peaks(data,mean+4*std,None,3)
    peak_locs = peak_tuple[0]
#    print peak_locs
    true_peaks = []
    for i in range(0,len(peak_locs)):
        p=peak_locs[i]
        h=peak_tuple[1]['peak_heights'][i]
#        print p,h
        if (p+2 < len(data)):
            xdata=range(p-1,p+2)
        else:
            xdata=range(p-1,len(data))
        ydata=data[p-1:p+2]
        guess=[p,1.2,h]
        try:
#            popt,pcov = opt.curve_fit(gaussian,xdata,ydata,guess,None,False,True,([0,.1,100],[1600,8,1000]))
            popt = opt.curve_fit(gaussian,xdata,ydata,guess,None,False,True,([-np.inf,0,-np.inf],[np.inf,np.inf,np.inf]))[0]
            true_peaks.append(popt[0])
        except:
            print "Failed " + str(p) + " at height " + str(h)
            continue
    out = []
    for p in peak_locs:
        out.append(p)
    
    return true_peaks

def vert_pix_peaks(imag, x, mean=100.0, std=15.0, yguess=120, tol = 8):
    img = imag.T
    data = img[x][yguess-30:yguess+30]
    peak_tuple = sig.find_peaks(data,mean+tol*std,None,5)
    peak_locs = peak_tuple[0]
    
    if(len(peak_locs)==0) : # Didn't find anything...
        return yguess
#    print peak_locs
    true_peaks = []
    for i in range(0,len(peak_locs)):
        p=peak_locs[i]
        h=peak_tuple[1]['peak_heights'][i]
#        print p,h
        if (p+5 < len(data)):
            xdata=range(p-5,p+5)
        else:
            xdata=range(p-5,len(data))
        ydata=data[p-5:p+5]
        guess=[p,13.0,h]
        try:
            popt = opt.curve_fit(gaussian,xdata,ydata,guess,None,False,True,([-np.inf,0,-np.inf],[np.inf,np.inf,np.inf])) [0]
            true_peaks.append(popt[0]+yguess-30)
        except:
            #print "Failed " + str(p) + " at height " + str(h)
            continue
            
    if (len(true_peaks)>1):
        truep=-1
        fit = 10000
        for i in true_peaks:
            if (abs(yguess - i) < fit):
                fit=abs(yguess-i)
                truep=i
        return truep
    elif (len(true_peaks)==0):
        return yguess
   # print x,true_peaks[0]
    else:
        return true_peaks[0]

########################## BEST-FIT PARAMETER GETTERS #########################

#functions which produce parameters for mappings from pixels to wavelengths
#("map_to_lambda") or from pixels to heights ("map_to_h")

def map_to_lambda(peaks,lines,order=2):
    '''Input: a "get_pix_peaks" array, the lines believed to correspond to each peak,
    and the y position at which this mapping is being done
    Output: the least-squares-fit map from x position to wavelength, fitting a 
    quadratic equation. Lambda = m1 * x^2 + m2 * x + b'''
    M = []
    for p in peaks:
        if(order==2):
            M.append([p**0,p**1,p**2])
        elif(order==1):
            M.append([p**0,p**1])
        else:
            print "Order must be 1 or 2 for map_to_lambda!"
            return 0,0,0
    params = lstsq(M,lines)
    b = params[0][0]
    if (order==2):
        m1 = params[0][2]
    else:
        m1=0
    m2 = params[0][1]
    return m1,m2,b

def map_to_h(peaks,order=3):
    '''takes a peak_vert_arr array and produces the parameters for a
    best fit polynomial of the order given. H = m1*x**2 + m1*x + b, or else
    H = m1*x + b if order = 1.
    '''
    if (order==3):
        M = []
        y = []
        for i in range(0,len(peaks)):
            p = peaks[i]
            M.append([i**0,i**1,i**2,i**3])
            y.append(p)
        params = lstsq(M,y)
        b = params[0][0]
        m3 = params[0][1]
        m2 = params[0][2]
        m1 = params[0][3]
        return m1,m2,m3,b
    if (order==2):
        M = []
        y = []
        for i in range(0,len(peaks)):
            p = peaks[i]
            M.append([i**0,i**1,i**2])
            y.append(p)
        params = lstsq(M,y)
        b = params[0][0]
        m2 = params[0][1]
        m1 = params[0][2]
        return 0,m1,m2,b
    elif (order==1):
        M = []
        y = []
        for i in range(0,len(peaks)):
            p = peaks[i]
            M.append([i**0,i**1])
            y.append(p)
        params = lstsq(M,y)
        b = params[0][0]
        m1 = params[0][1]
        return 0,0,m1,b
    else:
        print "Best fit must be order 1 or 2"
        return 0,0,0

############################ ARRAYS OF PEAK LOCATIONS #########################
#functions which produce arrays of peak locations, either per-row or per-column

def peak_arr(img):
    '''takes an image, returns an array detailing the locations of the peaks
    on each row
    '''
    peaks = []
    for i in range(0,len(img)):
        peaks.append(get_pix_peaks(img,i))
        #print "Row: "+str(i),
    return peaks

def peak_vert_arr(img,yguess=120,tol=8):
    '''takes an image, returns an array detailing the location of the peak
    on each column (assuming there is only on peak)
    '''
    peaks = []
    stats = st.sigma_clipped_stats(img)
    mean = stats[0]
    std = stats[2]
    for i in range(0,len(img[0])): #O(n)
        #print i,
        stats = st.sigma_clipped_stats(img.T[i])
        mean = stats[0]
        std = stats[2]
        peaks.append(vert_pix_peaks(img,i,mean,std,int(yguess),tol)) #O(n)
        yguess=peaks[i]
    return peaks

############################### "BEST_LINES" ##################################
#The "best_lines" function, for finding which lines a given row in your image
#probably represents, when calibrating for wavelength.

def best_lines(peaks,lines,m1=-0.00001,m2=-8.62,\
              b=8987, tol = 0.1,indexes=False):
    '''Takes a list of peaks and compares it to the line list given, and
    returns a best guess of which of the given lines are represented in the
    list of peaks. Recommended to be used iteratively. Lists will be sorted,
    to keep this function nearly O(n) rather than O(n^2), and to give usable
    indexes for index-matching mode. The outputs are two lists of the same 
    length: the first is the list of pixel peaks at which matches were found, 
    the second is the corresponding list of lines. It is recommend that the
    peaks given as "peaks" are sorted in advance, but this is not imposed.
    Optional parameters: a guess for the mapping parameters from pixels to 
    wavelengths. A tolerance, indicating how close a pixel peak has to be to
    a corresponding line to say "yes, that's THAT line," as a fraction. The
    "indexes" parameter is so that you can choose whether you want to know the
    value of the peaks and lines, or the indices of them, in your output.
    '''
    lines.sort()
    matched_peaks=[]
    matched_lines=[]
    lastLine = 0
    for x in range(0,len(peaks)):
        p = peaks[x]
        g = m1*p**2 + m2*p + b #the guessed wavelength for that pixel
        for num in range(0,len(lines)): 
            leng = len(lines)
            l1loc = lastLine+num #location of l1 in "lines"
            l1 = lines[l1loc%leng] #starts from the last match
            l0 = lines[(l1loc-1)%leng]
            l2 = lines[(l1loc+1)%leng]
            err1 = abs(l1-g)/l1
            err0 = abs(l0-g)/l0
            err2 = abs(l2-g)/l2
            if(err1<tol and err1<err0 and err1<err2):
                if (indexes):
                    alreadyfound = l1loc%leng in matched_lines
                else:
                    alreadyfound = l1 in matched_lines
                if (alreadyfound):
                    if (indexes):
                        pix = peaks[matched_peaks[len(matched_peaks)-1]]
                    else:
                        pix = matched_peaks[len(matched_peaks)-1]
                    old = m1*pix**2 + m2*pix + b
                    if (err1<(abs(l1-old)/l1)):
                        if(indexes):
                            matched_peaks[len(matched_peaks)-1]=x
                            matched_lines[len(matched_lines)-1]=l1loc%leng
                        else:
                            matched_peaks[len(matched_peaks)-1]=p
                            matched_lines[len(matched_lines)-1]=l1
                        lastLine=lastLine+num
                else:
                    if(indexes):
                        matched_peaks.append(x)
                        matched_lines.append(l1loc%leng)
                    else:
                        matched_peaks.append(p)
                        matched_lines.append(l1)
                    lastLine=lastLine+num
#                break
#            elif(err1>tol and lastLine+num): #stop looking if you've definitely passed it
#                print l1,g,"break"
#                break
    return matched_peaks,matched_lines


###############################################################################
#linelist reading function
    
def read_linelist(source):
    '''source file should be csv. I used a ".txt" file in which each line was
    separated by a comma, so it's the same thing.
    '''
    out=[]
    csvfile = open(source, 'r')
    csvreader = csv.reader(csvfile,delimiter=",")
    for row in csvreader:
        for item in row:
            i = float(item)
            out.append(i)
            #print item
    csvfile.close()
    return out

###############################################################################
#functions which produce 2d arrays (images) which are "1" where there is a peak,
#and "0" where there isn't one. Useful for visualizing if mappings aren't
#working, in both fits and with plt.imshow(...)

def binary_peaks(peaks,length=1530):
    ''' input: an array of, for our image, 1020(?) elements, each of which is an array
    listing the locations of the peaks on that row in the image.
    output: an image where every pixel is either 1 or 0, depending on whether or
    not there's a peak there
    '''
    newimg =[]
    for i in range(0,len(peaks)):
        newimg.append([0] * length)
#        print "adding peaks at y pixel: " + str(i)
        for p in peaks[i]:
            newimg[i][int(p)] = 1
    return newimg

def binary_peaks_binned(img):
    '''same as binary peaks but averages over rows 5 at a time
    '''
    peaks = binary_peaks(peak_arr(img))
    new_arr = []
    for i in range(0,len(img),5):
        sub_arr=[]
        for j in range(0,len(img[0])):
            add = peaks[i][j] + peaks[i+1][j] + peaks[i+2][j] + peaks[i+3][j] + peaks[i+4][j]
            add2 = add / 5.0
            if (add2 > 0.4):
                sub_arr.append(1)
            else:
                sub_arr.append(0)
        new_arr.append(sub_arr)
        new_arr.append(sub_arr)
        new_arr.append(sub_arr)
        new_arr.append(sub_arr)
        new_arr.append(sub_arr)
    return new_arr

def binary_vert_peaks(peaks,height=510):
    '''binary peaks, but vertical
    '''
    r = [0]*len(peaks)
    newimg = []
    newimg.append(r)
    newimg = newimg*height
    for i in range(0,len(peaks)):
        for p in peaks[i]:
            newimg[int(p)][i] = 1
            print i,int(p)
    return newimg

###############################################################################
#gaussian functions

def gaussian(x,m,var,a):
    denom = 1.0 / (np.sqrt(2.0*np.pi*var))
    expon = (0.0 - ((x-m)**2.0)) / (2.0 * var)
    numer = np.e ** expon
    out = denom*numer
    return a*out

def gaussian2d(x,y,mx,my,varx,vary,a):
    denom = 1.0 / (2.0 * np.pi * (varx*vary)**0.5)
    expon = -( ((x-mx)**2.0)/(2.0*varx) + ((y-my)**2.0)/(2.0*vary) )
    numer = np.e ** expon
    out = denom*numer
    return a*out

###############################################################################
#functions which actually produce image files    

def make_image(img,filename):
    '''img is a 2d array, like those produced by xy_to_lambda, xy_to_h, and the
    binary peaks functions.
    '''
    hdu = fits.PrimaryHDU(img)
    hdul = fits.HDUList([hdu])
    hdul.writeto(filename)

def make_peaks_image(img,filename):
    newimg = binary_peaks(peak_arr(img))
    make_image(newimg,filename)
    
def make_binned_image(img,filename):
    newimg = binary_peaks_binned(img)
    make_image(newimg,filename)
    






#def xy_to_lambdaQuick(slope = -40.857,r=75):
#    '''auto-calibrates, again based on the Argon and Neon lamps. This time, however,
#    the tilt is included manually. This is quicker, but potentially less accurate,
#    but also less susceptible to noise. A region of 5 pixels above the basis row, r,
#    are selected, and then the rest is extrapolated based on the slope. Same output
#    as regular lambdaAtEachXY.
#    '''
#    cal = np.asarray(CCDData.read('../Data/Arc-ArNe003.FIT', unit=1))
#    arr = []
#    lines = [7438.9,7245.2,7173.9,7032.4,6929.5,6717,6678.3,6599 \
#,6532.9,6506.5,6402.3,6383,6334.4,6304.8,6266.5,6217.3,6163.6,6143.1,6096.2,6074.3 \
#,6030,5975.5,5944.8,5881.9,5852.5,5790.7,5769.7,5460.7,4358,4047]
#    binnedRow=[]
#    for x in range(0,len(cal[r])):
#        pix = (cal[r][x]+cal[r+1][x]+cal[r+2][x]+cal[r+3][x]+cal[r+4][x]) / 5.0
#        binnedRow.append(pix)
#    peaks = get_pix_peaks(binnedRow)
#    if (len(peaks)==len(lines)):
#        m1,m2,b = map_to_lambda(peaks,lines)
#    elif (len(peaks)==(len(lines)+2)):
#        newLines = [7535.8,7488.9] + lines
#        m1,m2,b = map_to_lambda(peaks,newLines)
#    else:
#        print "Bad sample row!"
#        return []
#    for i in range(0,len(cal)):
#        row = []
#        for p in range(0,len(cal[i])):
#            x = p + (i-r)*(1.0 / slope)
#            l = m1 * x**2 + m2 * x + b
#            row.append(l)
#        arr.append(row)
#    return arr


#def getLambdaPeaks(img,y):
#    '''Lambda peaks are pixel peaks, but you use "map_to_lambda" to get the fit for
#    x and lambda at that y value, and then get wavelength peaks. You don't need to
#    know the tilt, because you can do this for each y value, and then round to the
#    nearest wavelengths. 
#    '''
#    pixPeaks = get_pix_peaks(img,y)
#    wavs = []
#    m1,m2,b = map_to_lambda(pixPeaks,lines)
#    for p in pixPeaks:
#        x = p - (y-240)*0.0244
#        l = 9684.69 - 4.31862 * x
#        wavs.append(l)
#    return wavs  

#def noiseArr(dataArr,fullData):
#    mean = np.mean(fullData)
#    std = np.std(fullData)
#    noise = []
#    for i in range(0,len(dataArr)):
#        if (dataArr[i] < mean+std):
#            noise.append(dataArr[i])
#    return noise



#img = np.asarray(CCDData.read('../Data/Arc-Ne001.FIT', unit=1)) #img is 1020 by 1530
    #to get the location of the maximum, say, use np.argmax. Divide that by 1530,
    #and mod it by 1020, and those are, respectively, the row and column of the max val
#arr = []
#i = 0
##j = 0
##plt.imshow(img)
#a = np.amax(img)
#ind = np.argmax(img)
#for x in img:
#    arr.append([])
#    for y in x:    
#        arr[i].append(y)
#    i=i+1
##plt.imshow(arr)
#arr2 = np.asarray(arr)
#plt.figure(1)
#plt.subplot(2,1,1)
#plt.imshow(arr,'inferno')
#
#print "Peaks at:\n"
#mean = np.mean(img)
#std = np.std(img)
#for x in range(0,1530):
#    if img[136][x] > (mean+2*std):
#        print str(img[136][x])+" at "+str(x)
#
#vertAverage = []
#imgSwapped = np.swapaxes(img,0,1)
#for x in imgSwapped:
#    vertAverage.append(np.mean(x))
#
#plt.subplot(2,1,2)
#plt.plot(vertAverage)
#plt.ylabel('counts')
#plt.show()
#
#print "Averaged peaks at:\n"
#mean = np.mean(vertAverage)
#std = np.std(vertAverage)
#for x in range(0,1530):
#    if vertAverage[x] > (mean+2*std):
#        print str(vertAverage[x])+" at "+str(x)
#        
#rotated = transform.rotate(img,1.4)
#
#vertAvRot = []
#imgSwapped = np.swapaxes(rotated,0,1)
#x = 30
#while x < (len(imgSwapped) - 30):
#    vertAvRot.append(np.mean(imgSwapped[x]))
#    x = x+1