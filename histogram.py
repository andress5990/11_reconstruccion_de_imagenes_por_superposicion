import cv2
import numpy as np
from matplotlib import pyplot as plt
import pdb


drawing = False # true if mouse is pressed
ix = -1 #Creamos un punto inicial x,y
iy = -1,
dotslist = [] #Creamos una lista donde almacenaremos los puntos del contorno

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
            cv2.line(img1, (x,y), (x,y), (0,0,0), 2)#Dibujamos una linea de un solo pixel
            x = x
            y = y
            dot = [x,y]
            dotslist.append(dot)#Agregamos el punto a dotslist
            #print(dotslist) #Imprimimos el dotslist

    elif event == cv2.EVENT_LBUTTONUP:#Cremaos el evento si el boton se levanta
        drawing = False
        #cv2.circle(img,(x,y),1,(0,0,255),1)
        cv2.line(img1, (x,y), (x,y), (0,0,0), 2)#Dibujamos la ultima lina en el ultimo punto

    return dotslist#Retornamos el dotlist



def Delet_Band_Name(img):
    img = img[40:407, 0:500]
    return img

def Croped(dotslist, img):#hacemos un corte de la imagen en linea recta de tal forma que tenga las
                          #dimenciones maximas del poligono que creamos
    rect = cv2.boundingRect(dotslist)#Encontramos los limites maximos del
    (x,y,w,h) = rect#Tomamos las dimenciones maximas del dotlist y las guardamos para dimencionar la mascara
    croped = img[y:y+h, x:x+w].copy()#cortamos una seccion rectangular de la imagen
    #dotslist2 = dotslist- dotslist.min(axis=0)#reajustamos el dotslist con el minimo

    mask = np.zeros(croped.shape[:2], dtype = np.uint8)# creamos una mascara de ceros para poder hacer el corte irregular
    cv2.drawContours(mask, [dotslist], -1, (255,255, 255), -1, cv2.LINE_AA)#dibujamos el contorno
    dts = cv2.bitwise_and(croped,croped, mask=mask)#hacemos ceros todos los pixeles externos al contorno

    return [dts, mask, croped]

img1 = cv2.imread('1-S.tif')
img2 = cv2.imread('1-C.tif', 0)

img1 = Delet_Band_Name(img1)#recortamos el encabezado
img2 = Delet_Band_Name(img2)#recortamos el encabezado

cv2.namedWindow('1-S-tif')#Cremaos la ventana para mistras a img
cv2.setMouseCallback('1-S-tif',draw_dots) #llamamos al MouseCall para dibujar el contorno

while(1):

    cv2.imshow('1-S-tif', img1) #Mostramos a img en la ventana para dibujar el contono
    img11 = img1.copy()


    k = cv2.waitKey(1) & 0xFF

    if k == 32: #space

        img11 = cv2.cvtColor(img11, cv2.COLOR_BGR2GRAY)
        dotslist = np.asarray(dotslist)#convertimos img1 en un np.array()

        img_croped_BB = Croped(dotslist, img2)[0]#Rcuperamos solo la region de interes (imagen cortada con bordes negros)
        mask = Croped(dotslist, img2)[1] #Recuperamos la mascara creada
        img_croped = Croped(dotslist, img2)[2]#recuperamos la imagen cortada en rectangulo para analizar con la mascara

        histb = cv2.calcHist([img11],[0],mask,[256],[0,256])
        histg = cv2.calcHist([img2],[0],None,[256],[0,256])
        plt.plot(histb, color ='b')
        plt.plot(histg, color ='g')
        plt.xlim([0,256])
        plt.show()


    if k == 27:#esc
        break


#pdb.set_trace()
