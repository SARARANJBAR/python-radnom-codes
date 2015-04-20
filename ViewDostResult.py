# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 14:59:49 2015

load IMAGE
calculate dost
plot Dost Images
plot harmonics image
@author: sara
"""

import sys
sys.path.append('/Users/sara/Dropbox/Texture Pipeline Source')

import os
from PIL import Image
import fnmatch
import DostFunctions as Dost
from string import maketrans
import matplotlib.pyplot as plt
from numpy import power,array,ceil,log2,sqrt,uint8


rootDir = '/Users/sara/anaconda/envs/py27/Examples/untitled folder'
targetDir = ''
targetImg = '*.gif'
targetROI = '*.txt'





def extractRectangle( image, coords ):
    cornerA = coords[0]
    cornerB = coords[1]
    subImage = image[ cornerA[0]:cornerB[0] , cornerA[1]:cornerB[1] ]
    return subImage
    
    
def convertROItopowerof2(coords):

    cornerA = coords[0]
    cornerB = coords[1]

    centerX = int((cornerB[0] + cornerA[0])/2.0)
    centerY = int((cornerB[1] + cornerA[1])/2.0)
   
    new_dx = power(2,ceil(log2(cornerB[0] - cornerA[0])))
    new_dy = power(2,ceil(log2(cornerB[1] - cornerA[1])))
    
    new_cornerA = (centerX - new_dx/2  , centerY - new_dy/2  )
    new_cornerB = (centerX + new_dx/2 , centerY + new_dy/2 )
 
    return [new_cornerA , new_cornerB]
    
    
def parseRectROIFile( fn ):
    
    with open( fn, "r" ) as infile:
        
        # skip the first 2 lines in the ROI file
        for i, line in enumerate( infile ):
            if i < 2: continue
        
        # use the string method translate() to replace the chars "(-)" anywhere
        # one of them appears in the input line, with a space.
        #
        # to do this we must first setup a char translation table
        intab = "(-)"
        outtab = "   "
        trantab = maketrans( intab,outtab )
    
        # replace open and close round brackets, and dashes, with spaces 
        clean = line.translate( trantab )

        # pick the numbers out of the clean input line
        nums = clean.split()
        coords = [ ( int(nums[0]), int(nums[1]) ), ( int(nums[2]), int(nums[3]) ) ] 

    return coords
    

def computeDostForallPixels(root, fnImg, fnROI):
    

    rootFnImg = os.path.join(root , fnImg)
    inputImage = array(Image.open( rootFnImg))

    rootFnROI = os.path.join(root, fnROI)
    rect = parseRectROIFile( rootFnROI )
    power2ROI = convertROItopowerof2(rect)
    subImage = extractRectangle( inputImage, power2ROI ).astype( float )

    dostImage = Dost.DOST(array(subImage))


    harmonicsImage = Dost.pointlocalSpectrum(dostImage,len(dostImage)/2,len(dostImage)/2)
    out = Image.fromarray(sqrt(abs(harmonicsImage)).astype(uint8))
    plt.figure(3)  
    plt.imshow(out,interpolation='nearest') 
    
    out2 = Image.fromarray(sqrt(abs( fftshift(harmonicsImage))).astype(uint8))
    plt.figure(4)  
    plt.imshow(out2,interpolation='nearest') 

    p = Dost.Generate_primary_freq_component(harmonicsImage)
    return p

for root, dirnames, filenames in os.walk( rootDir ):
    for fnImg in fnmatch.filter( filenames, targetImg ):
        print( " Processing Image:   ", fnImg )
        for fnROI in fnmatch.filter( filenames, targetROI ):
            computeDostForallPixels(root, fnImg, fnROI)
            
            