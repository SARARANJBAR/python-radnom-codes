#!/usr/bin/env python  
#=========================================================================
#
# Extract texture features from a region-of-interest within a dicom image
#
# This program expects two 2D dicom images as input
#   the first of these is a gray scale image containing the medical image
#   to be analyzed
#
#
#
# Output consists of ...  
#
#=========================================================================



from __future__ import print_function

import os
import sys
import dicom
import fnmatch
import mahotas
import mahotas.features

#import matplotlib.pyplot as plt
#import myshow

from string import maketrans

from numpy import *
from DostFunctions import*
from PIL import Image
from ImgFunctions import *


def getTestImage():
    testImage = np.array( [[0,0,1,1], [0,0,1,1], [0,2,2,2], [2,2,3,3]] )
    return testImage
    
        
    

def readDicomImage( fn ):
    #
    # load a dicom image. Does not handle jpeg2000 
    #
    #print( "Reading input image:", sys.argv[1] )
    df = dicom.read_file( fn )

    inputImage = df.pixel_array
    
    return inputImage

    #size = inputImage.GetSize()
    #print( "Image size:", size )

#
# extract and return a rectangular region from an image
#
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
    '''
    
     Parse an input file that looks something like this:
    
        256X256
        Rectangular ROI with 1 rectangle:
        (156 79)-(162 85)
    
     to extract the coordinates of the rectangle corners
    
     return the corners as a pair of 2D coordinates, in this example:
    
       [ (156,79) , (162,85) ]
    
    
    '''
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
    
 
def getDostFeatureNames(n):
    '''
    returns average of hormonics for an 8*8 ROI
    8*8 ROI gives a 5*5 harmonics Image
    we average to insure rotational invariance:
    
    F E D E F
    E C B C E
    D B A B D
    E C B C E
    F E D E F

    A : harmonics (0,0)
    B : harmonics pair (1,0),(0,1) 
    C : harmonics pair (1,1) : diagonal
    D : harmonics pair (2,0)
    E : harmonics pair (2,1),(1,2)
    F : harmonics pair (2,2) : diagonal
    '''
    # n is the length of the feature Vector
    a = 'Dost'
    featureNames = []
    for i in range(n):
        featureNames.append(a+str(i))

    return featureNames
    
    
def getGLCMFeatureNames():
    '''
    Return a list of feature names computed from the GLCM
    
    13 features are defined. Note:
    - Uniformity is also called Angular 2nd Moment (ASM, used here) , and Energy ** 2
    - Inverse Difference Moment is also called Homogeneity (used here)

    '''
    featureNames = [ 'ASM', 'Contrast', 'Correlation', 'Sum of Squares', 'Homogeneity', \
                     'Sum Avg', 'Sum Var', 'Sum Entropy', 'Entropy', 'Diff Var',  \
                     'Diff Entropy', 'Info Meas Corr 1', 'Info Meas Corr 2' ]
    return featureNames
    
#
# Calculate GLCM (Haralick) texture features for an image
#
def computeGLCMFeatures( image ):
    
    # calculate the GLCM for each of 4 directions, then calculate and return 13 texture
    # features for each direction. The 14th Haralick texture feature,
    # "Maximal Correlation Coefficient" is not calculated. 
    # The Mahotas documentation claims this feature is 'unstable' and should not be used...
    f = mahotas.features.haralick( image )
    
    # calculate the mean feature value across all 4 directions
    fMean = f.mean( 0 )
    
    # calculate the range (peak-to-peak, ptp) of feature values across all 4 directions
    fRange = f.ptp( 0 )
    
    # 13 features are returned
    # Uniformity is also called Angular 2nd Moment (ASM, used here) , and Energy ** 2
    # Inverse Difference Moment is also called Homogeneity (used here)
    featureNames = getFeatureNames()
        
    # create an empty dictionary to hold the feature name, mean and range values             
    d = {}
    
    # fill each dictionary entry, with the name (text), followed by that feature's mean and range (a tuple of floats)
    for i, name in enumerate( featureNames ):
        d[ name ] = ( fMean[ i ], fRange[ i ] )
        
    return d

    
def getGLCMFeaturesFromROI( root, fnImg, fnROI, grayScales ):
    
    rootFnImg = os.path.join(root, fnImg)
    rootFnROI = os.path.join(root, fnROI)
    
    #inputImage = readDicomImage( rootFnImg )
    inputImage = Image.open(rootFnImg)
    #plt.imshow( inputImage, cmap=plt.cm.gray )

    #print( "Parsing ROI file:", sys.argv[2] )
    #mask = [ (100, 100), (128, 128) ]
    rect = parseRectROIFile( rootFnROI )
    
    # normalize the gray scale values in the ROI to range from 0 to grayScales
    tmp = extractRectangle( inputImage, rect ).astype( float )
    subImage = np.rint( grayScales * ( ( tmp - tmp.min() ) / tmp.ptp() ) ).astype( int )
    plt.imshow( subImage, cmap=plt.cm.gray )
    
    #subImage = getTestImage()
    
    # calculate GLCM texture features for the rectangular sub image
    features = computeGLCMFeatures( subImage )
    features[ "ROI Mean" ] = ( tmp.mean(), tmp.ptp() )
    features[ "ROI Variance" ] = ( tmp.var(), 0 )
    
    # now add a tuple (pair) of strings to the feature list. These
    # provide info on the patient, image and ROI that produced the features
    features[ "0_File Info" ] = ( fnImg , fnROI )
  
    return features
      

def writeHeader( outFile, hdr, d ):
    '''
    
     Create a file containing a header followed by a row of feature names
    
     input consists of an output file object, a header (dictionary), followed by a dictionary of feature names. 
     Associated with each feature name is the feature mean value, and feature range
    
    for example, suppose your feature dictionary contains the following:     
    
       { 'Ang 2nd Moment': (1.0, 2.0), 
         'Entropy': (3.0, 4.0),
         'Correlation': (5.0, 6.0) }
    
     Then the output would look like:
    
       Ang 2nd Moment      Entropy         Correlation
       Mean    Range       Mean  Range     Mean    Range
       1.0     2.0         3.0   4.0       5.0     6.0
    
    
    '''
   
    for name in sorted( hdr ):
        outFile.write( "%s:\t" % name )
        outFile.write( "%s\n" % hdr[name] )
    outFile.write( "\n" )

    for name in sorted( d ):
        outFile.write( "%s\t" % name )
    outFile.write( "\n" )
        

 
def writeFeatures( outFile, d ):
    '''
     Add a row of features to the output file.
     Features are stored in a dictionary, d. Each dictionary entry (key)
     references a tuple (pair) of values. These may be of type string
     or float. The dictionary entries are sorted alphabetically by key (name)
     prior to output.
    
    '''   
    for name in sorted( d ):
   #     ty = type( d[name][0] )
        ty = type( d[name] )
        if ( ty is str ):
            outFile.write( "%s\t" % d[ name ])           
        else:
            outFile.write( "%.2f\t" % d[ name ] )

    outFile.write( "\n" )
 
 
def computeDostfeaturesForImageCenter(root, fnImg):
    
    rootFnImg = os.path.join(root, fnImg)
    inputImage = readDicomImage( rootFnImg ) 
    dostImage = DOST(array(inputImage))
    harmonicsImage = localSpectrum(dostImage,len(dostImage)/2,len(dostImage)/2)  
    f = calculate_features(real( harmonicsImage ))
    featureNames = getDostFeatureNames(len(f))
              
    d = {}
    for i, name in enumerate( featureNames ):
        d[ name ] = ( f[i] )
      
    return d

def computeDostFeaturesForROIcenter(root, fnImg, fnROI):
    
    tmp = fnImg.strip('.png')
#    tmp = tmp.strip('Test_freq')
#    fcontent, ncontent = tmp.split("_", 1)
#    sdnr = ncontent.strip('sdnr')
    
   
    
    rootFnROI = os.path.join(root, fnROI)

    inputImage = readImage( root, fnImg ) 
#    rect = parseRectROIFile( rootFnROI )
#    power2ROI = convertROItopowerof2(rect)
#    subImage = extractRectangle( inputImage, power2ROI ).astype( float )

#    fig = plt.figure(1)
#    ax = fig.add_subplot(111)
    dostImage = DOST(array(inputImage))
#    plt.imshow(Image.fromarray(sqrt(abs(dostImage)).astype(np.uint8)),interpolation = 'nearest')
#    n=len(dostImage)/2 
#    x = np.arange(-n,n+1)
#    labels = [str(int(i)) for i in x]
#    ax.set_xticklabels(labels)
#    start, end = ax.get_xlim()
#    ax.xaxis.set_ticks(np.arange(start+0.5, end, 1))
#    y=-x
#    labels = [str(int(i)) for i in y]
#    ax.set_yticklabels(labels)
#    ax.yaxis.set_ticks(np.arange(start+0.5, end, 1))
#    plt.show()
    
    
#    fig = plt.figure(2)
#    ax = fig.add_subplot(111)
    harmonicsImage = pointlocalSpectrum(dostImage,len(dostImage)/2,len(dostImage)/2)
#    plt.imshow(Image.fromarray(sqrt(abs(harmonicsImage)).astype(np.uint8)),cmap=plt.cm.gray, interpolation = 'nearest')
#    n=len(harmonicsImage)/2 
#    x = np.arange(-n,n+1)
#    labels = [str(int(i)) for i in x]
#    ax.set_xticklabels(labels)
#    start, end = ax.get_xlim()
#    ax.xaxis.set_ticks(np.arange(start+0.5, end, 1))
#    y=-x
#    labels = [str(int(i)) for i in y]
#    ax.set_yticklabels(labels)
#    ax.yaxis.set_ticks(np.arange(start+0.5, end, 1))
#    plt.title(tmp)
#    plt.show()
    
    

    p = Generate_primary_freq_component(harmonicsImage)
    print("primary frequency",p)
    
    graph = Generate_graph(harmonicsImage)
#    print(graph.)
    
    plt.show()
    f = calculate_features( real( harmonicsImage ))
#    featureNames = getDostFeatureNames(len(f))
    
    features = {}

#    for i, name in enumerate( featureNames ):  
#         features[ name ] = ( f[i] )
#   
#    tmp = fnImg.strip('.dcm')
#    tmp = tmp.strip('Test_freq')
##    fcontent, ncontent = tmp.split("_", 1)
#    features[ "ROI size" ] =  subImage.shape[0]
##    ncontent = ncontent.strip('sdnr')
#    features[ "Image Name" ] =  fnImg   
#    features[ "sdnr"] = 'inf'  
#    features[ "InputFrequency" ] =  tmp 
  
    return features
    
def computeandwriteDostForallPixels(root, fnImg, fnROI, outFile):
    
    tmp = fnImg.strip('.dcm')
    tmp = tmp.strip('Test_freq')
    fcontent, ncontent = tmp.split("_", 1)
    Sigma = ncontent.strip('SD')
    
    rootFnImg = os.path.join(root, fnImg)
    inputImage = readDicomImage( rootFnImg )    

    rootFnROI = os.path.join(root, fnROI)
    rect = parseRectROIFile( rootFnROI )
#    power2ROI = convertROItopowerof2(rect)
#    subImage = extractRectangle( inputImage, power2ROI ).astype( float )
    dostImage = DOST(array(inputImage))

    features = {}
    
    x0, y0  = rect[0][0], rect[0][1]
    x1, y1  = rect[1][0], rect[1][1]
    
    
    for x in range(x0,x1):
        for y in range(y0,y1):
            harmonicsImage = pointlocalSpectrum(dostImage,x,y)
#            f = calculate_features(real( harmonicsImage ))
#            featureNames = getDostFeatureNames(len(f))
#            for i, name in enumerate( featureNames ):  
#                 features[ name ] = ( f[i] )
                 
            p = Generate_primary_freq_component(harmonicsImage)
            Nu = convertHarmonicsIndexTovoiceFrequencies(p)
            features ["Px "] = p[0]
            features ["Py "] = p[1]
            features ["Detected Voice_x Frequency"] = Nu[0]
            features ["Detected Voice_y Frequency"] = Nu[1]
            features[ "Image Name" ] =  fnImg
            features[ "Coordinate x " ]  = x
            features[ "Coordinate y" ]  = y
            features[ "SD"] = Sigma  
            features[ "InputFrequency" ] =  fcontent 
            

            writeFeatures( outFile, features )
         
    return features    

def computeaverageDostFeaturesforROI( root, fnImg, fnROI):
    
    rootFnImg = os.path.join(root, fnImg)
    inputImage = readDicomImage( rootFnImg )    

    rootFnROI = os.path.join(root, fnROI)
    rect = parseRectROIFile( rootFnROI )
    power2ROI = convertROItopowerof2(rect)
    
    #            maskImage  = np.zeros((len(inputImage),len(inputImage)))
    maskImage = readDicomImage( rootFnImg )
    for x in range(rect[0][0],rect[1][0]):
        for y in range(rect[0][1],rect[1][1]):
            maskImage[y][x] *=6
            
    plt.figure(figsize= (10,10))        
    plt.imshow(maskImage ,cmap=plt.cm.gray)
    plt.title("mask Image")
    plt.show()
    
    
    subImage = extractRectangle( inputImage, power2ROI ).astype( float )
    dostImage = DOST(array(subImage))
    harmonicsImage = averagelocalSpectrum( dostImage )
    f = calculate_features(real( harmonicsImage ))
    features = {}
    featureNames = getDostFeatureNames(len(f))
    for i, name in enumerate( featureNames ):  
        features[ name ] = ( f[i] )
    
    features[ "Image Name" ] =  fnImg 
    tmp = fnROI.strip('.txt')
    roi_size_x,roi_size_y = tmp.split("*", 1)
    features[ "ROI size" ] =  subImage.shape[0]
    
    
    tmp = fnImg.strip('.dcm')
    tmp = tmp.strip('Test_freq')
#    fcontent, ncontent = tmp.split("_", 1)
    
#    ncontent = ncontent.strip('sdnr')
 #   sdnr,slicenumber = ncontent.split("_", 1)

 #   print(fcontent,ncontent)
    features[ "SDNR"] = 'inf'
    features[ "InputFrequency" ] =  tmp 
    
    return features

#
# main()
#
# Please update the versionInfo variable when substantial changes are made
#

versionInfo = "0.2b Jan 28 2015"
    
if len ( sys.argv ) < 2:
    print( "Usage: extractTextureFeatures <root_dir> <output_file.txt>" )
    
    # normally one would just exit here. The following 2 lines let the program run
    # from the debugger, without command line input. Change the root path to meet your 
    # degub needs
    fnOut = "out.txt"
    rootDir = '/Users/sara/anaconda/envs/py27/Examples/untitled folder'
#    sys.exit ( 1 )
else:
    rootDir = sys.argv[1]
    fnOut = sys.argv[2]

# identify the target directory name, image files (dicom), and ROI files
targetDir = '[Ss]lice*'
targetImg = '*.png'
targetROI = '*.txt'
outFile = open( fnOut, "w" )
fileCount = 0
grayScales = 256

# create a blank dictionary to hold header info
hdr = {}
#hdr[ "Gray Scales" ] = grayScales
#hdr[ "Entropy" ] = "log2"

# use a dumb trick to make the title appear first in the sorted output
#hdr[ "0_Title" ] = "GLCM Features"
hdr[ "0_Title" ] = "DOST Features"
hdr[ "Version" ] = versionInfo
hdr["1_comment"] = " images: no noise, known frequency content, ROIs: 16*16 at the center, point Dost features computed for the center of each partition"

features = {}
# recursibley search starting at the root directory...
# we are looking for files that match targetROI
# once found, we then iterate over all files in that dir that match targetImg
# 
# Run the texture feature extractor on the ROI within an image file
for root, dirnames, filenames in os.walk( rootDir ):
    for fnROI in fnmatch.filter( filenames, targetROI ):
        print( "Processing ROI:", fnROI )
        for fnImg in fnmatch.filter( filenames, targetImg ):
            if( not fileCount ):
                writeHeader( outFile, hdr, features )      
            print( " Processing Image:   ", fnImg )
       #     features = getGLCMFeaturesFromROI( root, fnImg, fnROI, grayScales )
       #     features = computeandwriteDostForallPixels(root, fnImg, fnROI,outFile)  # had to combine compute and write, dictionary isnot expandable
       #     features = computeandwriteDostForallPixels(root,fnImg,fnROI,outFile)   
       #     features = computeaverageDostFeaturesforROI(root,fnImg,fnROI)
            features = computeDostFeaturesForROIcenter(root,fnImg,fnROI)
            writeFeatures( outFile, features )
            fileCount = fileCount + 1
           
        outFile.flush()

#if( not fileCount ):
      
print( "Input Count: ", fileCount )
#writeHeader( outFile, hdr, features )           
outFile.close()
         
    



