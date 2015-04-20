# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 10:57:28 2015

@author: sara
"""

import xlrd as xlrd
import matplotlib.pyplot as plt
from numpy import meshgrid,array,append,linspace,uint8,transpose
from PIL import Image

filename = '/Users/sara/Dropbox/sdnr40.xlsx'

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

Y =  array([4,8,16,32,64])

X, Y = meshgrid(X,Y)

Z=[]
for i in range(len(f4)):
    Z= append(Z,array([f4[i],f8[i],f16[i],f32[i],f64[i]]))
print(Z)    
Z1 = Z.reshape((26,5))
Z2 = transpose(Z1)

fig = plt.figure()
ax = fig.add_subplot(111)
out = Image.fromarray(Z2.astype(uint8))

plt.xticks(append(array([1]),linspace(10, 250, num=25)))
plt.yticks(array([4,8,16,32,64]))
plt.imshow(out,interpolation = 'nearest', cmap=cm.hot)
plt.colorbar(shrink = 0.25)  
fig.suptitle('average Dost Features sdnr = 40, Diagonal Dost features included')
plt.show()
