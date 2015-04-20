# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 12:50:57 2015

@author: sara
"""

from xlrd import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
import numpy as np
from PIL import Image

filename = '/Users/sara/Desktop/sylviaMatlab.xlsx'

workbook = open_workbook(filename)
worksheet=workbook.sheet_by_index(0)
   
InputFreq = worksheet.col_values(colx=3)
Sigma = worksheet.col_values(colx=4)
px = worksheet.col_values(colx=5)
py = worksheet.col_values(colx=6)


frequencies = np.append(np.linspace(1,9,9),np.linspace(10,120,12))
StandardDeviation = np.linspace(0,1,11)
Length = frequencies.size

percentages = []

BulkSize = 100


iterator = np.linspace(0,99,100).astype(int)

begin_index = 0 

counts = frequencies.size * StandardDeviation.size

logF = [np.ceil(np.log2(x))  for x in InputFreq]


for ii in range(0,counts):

        end_index = begin_index + BulkSize
       
        Val = logF[begin_index:end_index] 
        Px = px[begin_index:end_index] 
        Py = py[begin_index:end_index]
        
        
        
        c = [ 1 if (Py[i]==Val[i] and Px[i]==Val[i]) else 0  for i in iterator]
        percentages = np.append(percentages,np.array([sum(c)]))
        begin_index = end_index 
        
   
        
X, Y = np.meshgrid(frequencies,StandardDeviation)
Z    = percentages.reshape((frequencies.size,StandardDeviation.size))  
Z = np.transpose(Z) 
print Z      
        


fig = plt.figure()
ax = fig.add_subplot(111)
out = Image.fromarray(Z.astype(np.uint8))
plt.imshow(out,interpolation = 'nearest', cmap=cm.jet)
plt.colorbar(shrink = 0.25)