import cv2
import numpy as np
import sys
import math

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
#style.use('fivethirtyeight')

from PIL import Image

import pdb

#def convert_tiffTojpeg(images):
#    images_names = []
#    #pdb.set_trace()
#    for name in images:       
#        im = Image.open(name)
#        im.save(name + '.jpg', 'JPEG')
#        image_name = name.replace('.tif', '.jpg')
#        images_names.append(image_name)
#      
#    #pdb.set_trace()
#    return images_names

def cal_luminosity(file_name):
    
    cv_img = cv2.imread(file_name)#np.array con shape(filas, columnas, canales)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB) #volvemos el valor del np.array a RGB para procesar
    cv_img = cv_img[40:407, 0:500]
    #cv_img = cv2.GaussianBlur(cv_img, (5,5), 0)
    #b, g, r = cv2.split(cv_img)#separamos los canales en gbr
    b = cv_img[:,:,0]#Estas lineas hacen lo mismo que la anterior, pero mas eficiente
    g = cv_img[:,:,1]
    r = cv_img[:,:,2] 
    
    #Este for hace lo mismo que cv_img2 = cv2.imread(file_name, 0)
    holder = np.ones(cv_img[:,:,0].shape)
    for i in range(b.shape[0]):#Filas
        for j in range(b.shape[1]):#Columnas
            lum=0.2126*r[i][j] + 0.7152*g[i][j] + 0.0722*b[i][j] #Photometric/digital ITU BT.709
            #lum=0.299*r[i][j] +0.587*g[i][j]+0.114*b[i][j] #Digital ITU BT.601 
                                                            #(gives more weight to the R and B components)
            holder[i][j] = lum
    
    #Normalizamos la matriz
    
    maxholder = np.amax(holder)
    for i in range(b.shape[0]):
        for j in range(b.shape[1]): 
            holder[i][j] = holder[i][j]/maxholder
    
    return(holder, maxholder)

def cal_ratio(img1, img2, row, column, cv_img1max, cv_img2max):
    
    #img1 = cv2.GaussianBlur(img1, (7,7), 0)
    #img2 = cv2.GaussianBlur(img2, (7,7), 0)
 
    holder = np.ones((row, column))

    #pdb.set_trace()

    for i in range(holder.shape[0]):
        for j in range(holder.shape[1]):
            
            if cv_img2max > cv_img1max: 
                a = img1[i][j]/img2[i][j]
                holder[i][j]= a
                #if a == np.nan or a == float('inf'):
                #    holder[i][j] = img1[i][j] 
                #if a != np.nan or a != float('inf'):
                #    holder[i][j] = a 

            elif cv_img1max > cv_img2max:
                a = img2[i][j]/img1[i][j]
                holder[i][j] = a
                #if a == np.nan or a == float('inf'):
                #    holder[i][j] = img2[i][j]
                #if a != np.nan or a != float('inf'):
                #    holder[i][j] = a
    #holder = cv2.GaussianBlur(holder, (5,5), 0)
    return holder
#--------------------------------------------------------------------------------------------------            

#files_names = [sys.argv[1], sys.argv[2]]
#images_names = convert_tiffTojpeg(files_names)
images_names = [sys.argv[1], sys.argv[2]]
#pdb.set_trace()

#creamos ambas matrices a partir de del calculo de luminosidad (grayscale)
cv_img1 = cal_luminosity(images_names[0])[0]
cv_img1max = cal_luminosity(images_names[0])[1]#Son de las imagenes no normalizadas
cv_img2 = cal_luminosity(images_names[1])[0]
cv_img2max = cal_luminosity(images_names[1])[1]#Son de las imagenes no normalizadas

row = len(cv_img1[:,1])
column = len(cv_img1[1,:])
#pdb.set_trace()

#Creamos la imagen con la razon de las otras dos
cv_result = cal_ratio(cv_img1, cv_img2, row, column,
                      cv_img1max, cv_img2max)

#pdb.set_trace()

cv_result_min = np.nanmin(cv_result)
cv_result_max = np.nanmax(cv_result)
#pdb.set_trace()

bounds = np.linspace(cv_result_min, cv_result_max, 10, endpoint=True)

ax1 = plt.subplot(111)
pltimg = ax1.imshow(cv_result, cmap = plt.cm.jet)
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(pltimg, cax = cax, ticks = bounds)

plt.show()


#Plot de color para la matriz resultante

#pil_cv_img1 = Image.fromarray(cv_img1)
#pil_cv_img2 = Image.fromarray(cv_img2)
#pil_cv_result = Image.fromarray(cv_result)

#pil_cv_img1.show()
#pil_cv_img2.show()
#pil_cv_result.show()