import cv2
import numpy as np
import sys
import math

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar

import pdb

drawing = False # true if mouse is pressed
ix = -1 #Creamos un punto inicial x,y
iy = -1,
dotslist = [] #Creamos una lista donde almacenaremos los puntos del contorno

def Listing(y):
    y1 = []#Creamos una lista vacia
    for i in range(len(y)):#llenamos la lista vacia con los datos de y, esto porque y es de la forma y = [[],[],[]], y necesitamos y = []
        y1.append(y[i][0])  
        
    return y1


def draw_dots(event,x,y,flags,param): #Crea los puntos de contorno
    global ix,iy,drawing, dotslist#Hacemos globales la variabbles dentro de la funcion

    if event == cv2.EVENT_LBUTTONDOWN:#creamos la accion que se realizara si damos click
        drawing = True #Drawinf se vuelve True
        ix = x #Tomamos el punto donde se dio click
        iy = y
        dot = [x,y]
        dotslist.append(dot)#Lo agregamos al dotslist

    elif event == cv2.EVENT_MOUSEMOVE:#Creamos la accion si el mouse se mueve
        if drawing == True: #drawing se vuelve true
            #cv2.circle(img,(x,y),1,(0,0,255),2)
            cv2.line(img1, (x,y), (x,y), (255,255,255), 2)#Dibujamos una linea de un solo pixel
            x = x
            y = y
            dot = [x,y]
            dotslist.append(dot)#Agregamos el punto a dotslist
            #print(dotslist) #Imprimimos el dotslist

    elif event == cv2.EVENT_LBUTTONUP:#Cremaos el evento si el boton se levanta
        drawing = False
        #cv2.circle(img,(x,y),1,(0,0,255),1)
        cv2.line(img1, (x,y), (x,y), (255,255,255), 2)#Dibujamos la ultima lina en el ultimo punto
      
    return dotslist#Retornamos el dotlist


def Dark(dotslist, img, mask, x, y, w, h):#hacemos un corte de la imagen en linea recta de tal forma que tenga las 
                          #dimenciones maximas del poligono que creamos
    #croped = img[y:y+h, x:x+w].copy()#cortamos una seccion rectangular de la imagen
    dotslist2 = dotslist- dotslist.min(axis=0)#reajustamos el dotslist con el minimo 
    cv2.drawContours(mask, [dotslist2], -1, (255,255, 255), -1, cv2.LINE_AA)#dibujamos el contorno
    dts = cv2.bitwise_and(img,img, mask=mask)#hacemos ceros todos los pixeles externos al contorno
    
    return dts

def mask(dotslist, img):

    rect = cv2.boundingRect(dotslist)#Encontramos los limites maximos del
    (x, y, w, h) = rect#Tomamos las dimenciones maximas del dotlist y las guardamos para dimencionar la mascara
    #pdb.set_trace()
    croped = img[y:y+h, x:x+w].copy()
    mask = np.zeros(croped.shape[:2])
    
    return [mask, x, y, w, h]

def cal_luminosity(img):
    
    cv_img = img[40:407, 0:500]

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
   
    minholder = np.amin(holder)
    maxholder = np.amax(holder)    

    return[holder, maxholder, minholder]



files = [str(sys.argv[1]), str(sys.argv[2])]

img1 = cv2.imread(files[0])#Lee la imagen a color
cv2.namedWindow(files[0])#Cremaos la ventana para mistras a img
cv2.setMouseCallback(files[0],draw_dots) #llamamos al MouseCall para dibujar el contorno

img2 = cv2.imread(files[1])#Lee la imagen pero en intensidad (B and W)

while(1):
    cv2.imshow(files[0], img1) #Mostramos a img en la ventana para dibujar el contono

    k = cv2.waitKey(1) & 0xFF
    
    if k == 32: #space
        dotslist = np.asarray(dotslist)
        #pdb.set_trace()
        
        img1 = cal_luminosity(img1)
        img2 = cal_luminosity(img2)
        
        #Recuperamos la mascara
        mask = mask(dotslist, img1)[0]
        x_img = mask(dotslist, img1)[1]
        y_img = mask(dotslist, img1)[2]
        w_img = mask(dotslist, img1)[3]
        h_img = mask(dotslist, img1)[4]
        
        #Aplicamos el contorno a la image a partir de dtolist
        img1_Dark = Dark(dotslist, img1, mask, x_img, y_img, w_img, h_img)#Rcuperamos solo la region de interes (imagen cortada con bordes negros)    
     

    if k == 27:#esc
        #pdb.set_trace()
        break# hace,el break para para el programa si se estripa esc

cv2.destroyAllWindows()#Destruimos todas las ventanas