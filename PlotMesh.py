# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 10:57:28 2015

@author: sara
"""

import xlrd as xlrd
import matplotlib.pyplot as plt
from numpy import meshgrid,array,append,linspace,asarray


filename = '/Users/sara/Dropbox/dost-noise experiment/round8/Workbook1.xlsx'

workbook = xlrd.open_workbook(filename)
worksheet=workbook.sheet_by_index(0)
   
inputdata = worksheet.col_values(colx=0)
sdnr = worksheet.col_values(colx=1)
f4 = worksheet.col_values(colx=2)
f8 = worksheet.col_values(colx=3)
f16 = worksheet.col_values(colx=4)
f32 = worksheet.col_values(colx=5)
f64 = worksheet.col_values(colx=6)

X =  append(array([1]),linspace(10, 250, num=25))
Y =  append(array([1]),linspace(5, 50, num=10))
X, Y = meshgrid(Y,X)

Z = asarray(f4).reshape((26,11))


fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')
p = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.winter)

cb = fig.colorbar(p, shrink = 0.5)

fig.suptitle(' predicted frequency based on point Dost features. 4*4 ROI .Diagonal included')
ax.set_ylabel('input Freqency ')
ax.set_xlabel('Signal difference to noise')
ax.set_zlabel('output Frequency')
plt.show()
