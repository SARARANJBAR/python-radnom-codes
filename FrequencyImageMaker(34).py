# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 20:08:22 2015

@author: rmitch
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from PIL import Image


def myshow(img, title=None, margin=0.05, dpi=80 ):

    spacing = [ 1,1 ]
    ysize = img.shape[0]
    xsize = img.shape[1]

 
    # Make a figure big enough to accomodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    figsize = (1 + margin) * ysize / dpi, (1 + margin) * xsize / dpi
    def callback(z=None):

        extent = (0, xsize*spacing[1], ysize*spacing[0], 0)

        fig = plt.figure(figsize=figsize, dpi=dpi)

        # Make the axis the right size...
        ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

        #plt.set_cmap("gray")

        if z is None:
            ax.imshow(img,extent=extent,interpolation=None)
        else:
            ax.imshow(img[z,...],extent=extent,interpolation=None)

        if title:
            plt.title(title)

        plt.show()

    callback()



def createSinImage( size, freq, power ):
    
    # create a Fourier domain, then fill it with power at a particular frequency,
    # and also set the DC component to the image power.
    # inverse transform the domain to create a sinusoidal image
    #
    fimg = np.zeros( size, dtype=np.complex )    
    fimg[ freq ] = power    
    fimg[ (0,0) ] = power    
    img = np.real( np.fft.ifft2( fimg ) )
    
    return img
    
def addGaussianNoiseToImage(img , SD):

    mu, sigma = 0, SD  #mean and standard deviation of noise
    
    noiseImg = np.random.normal(mu,sigma, img.shape)
    
    final = img + noiseImg
    
    return final
    
def addUniformNoiseToImage( img, sdnr ):

    # sdnr is the signal-difference-to-noise ratio
    # assume the signal difference is equal to the max intensity - min intensity
    # Then, sdnr = img.ptp() / noise. 
    # Therefore, noise = img.ptp() / sdnr
    noise = img.ptp() / sdnr
    
    # generate a random image with mean = 0.0, range = 1.0, and std dev = 0.28
    # (all values approximate)
    noiseImg = np.random.uniform( low=-0.5, high=0.5, size=img.shape )
    
    # set the std dev of the noise image to 'noise', while keeping mean = 0.0
    noiseImg = (noiseImg - noiseImg.mean()) * (noise / noiseImg.std())
#    print(noiseImg.mean(),noiseImg.std())
    # add the noise image to the passed image and return the result
    final = img + noiseImg
    
    return final
    


    
def scaleIntensity( image, oMin, oMax ):
    # normalize the gray scale values in the image to range from iMin to iMax

    # range of intensities in the input image
    iRange = image.ptp()
    if iRange == 0:
        return -1

    iMin = image.min()

    oRange = oMax - oMin
    
    image -= iMin        
    fImage = image * ( float(oRange) / iRange ) + oMin

    return fImage


    
# main()
f = 1
freq = (f,f)
size = ( 256,256 )
power = 100000
#omin = 448
#omax = 575
omin = 224
omax = 287


sdnr = 40
nfiles = 1
output_dir = "/Users/sara/Desktop/GaussianNoiseImages"

fileCount = 0
SD = 0.0
for f in range(10,130,10):
    freq = (f,f)
    fimg = createSinImage( size, freq, power )
    for i in range( nfiles ): 
           for SD in range(1,11,1):
            #   noisyImg = addNoiseToImage( fimg, sdnr )
            sigma = float(SD)/10.0
            noisyImg = addGaussianNoiseToImage(fimg , sigma)
            img = np.rint( scaleIntensity( noisyImg , omin, omax ) ).astype( np.uint16 )
            oimg = sitk.GetImageFromArray( img )
            output_file = "Test_freq%03d_SD%.1f.dcm" % ( f, sigma  )
            fullOutputFn = os.path.join(output_dir, output_file)
    
            sitk.WriteImage( oimg,fullOutputFn )
            print( output_file )
  




#myshow( img1, title='1' )
#myshow( img2, title='2' )
#myshow( diff, title='diff' )

#test = sitk.ReadImage( 'test.dcm' )

