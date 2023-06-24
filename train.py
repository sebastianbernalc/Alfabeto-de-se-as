import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os


cap = cv2.VideoCapture(0) # 0 para la camara web
detector = HandDetector(maxHands = 1) # maxHands = 1 para detectar una sola mano

offset = 15 # margen de la imagen
imgSize = 300 # tamaño de la imagen

# obtiene la ruta de la carpeta actual
current_directory = os.getcwd() #Directorio actual del proyecto
folder_A= os.path.join(current_directory, "Detector", "data", "A") #Carpeta de la letra A
folder_B= os.path.join(current_directory, "Detector", "data", "B") #Carpeta de la letra B
folder_C= os.path.join(current_directory, "Detector", "data", "C") #Carpeta de la letra C
folder_D= os.path.join(current_directory, "Detector", "data", "D") #Carpeta de la letra D
folder_E= os.path.join(current_directory, "Detector", "data", "E") #Carpeta de la letra E
folder_F= os.path.join(current_directory, "Detector", "data", "F") #Carpeta de la letra F
folder_G= os.path.join(current_directory, "Detector", "data", "G") #Carpeta de la letra G
folder_H= os.path.join(current_directory, "Detector", "data", "H") #Carpeta de la letra H
folder_I= os.path.join(current_directory, "Detector", "data", "I") #Carpeta de la letra I 
folder_J= os.path.join(current_directory, "Detector", "data", "J")  #Carpeta de la letra J
folder_K= os.path.join(current_directory, "Detector", "data", "K") #Carpeta de la letra K
folder_L= os.path.join(current_directory, "Detector", "data", "L") #Carpeta de la letra L
folder_M= os.path.join(current_directory, "Detector", "data", "M") #Carpeta de la letra M
folder_N= os.path.join(current_directory, "Detector", "data", "N") #Carpeta de la letra N
folder_O= os.path.join(current_directory, "Detector", "data", "O") #Carpeta de la letra O
folder_P= os.path.join(current_directory, "Detector", "data", "P") #Carpeta de la letra P
folder_Q= os.path.join(current_directory, "Detector", "data", "Q") #Carpeta de la letra Q
folder_R= os.path.join(current_directory, "Detector", "data", "R") #Carpeta de la letra R
folder_S= os.path.join(current_directory, "Detector", "data", "S") #Carpeta de la letra S
folder_T= os.path.join(current_directory, "Detector", "data", "T") #Carpeta de la letra T
folder_U= os.path.join(current_directory, "Detector", "data", "U") #Carpeta de la letra U
folder_V= os.path.join(current_directory, "Detector", "data", "V") #Carpeta de la letra V
folder_W= os.path.join(current_directory, "Detector", "data", "W") #Carpeta de la letra W
folder_X= os.path.join(current_directory, "Detector", "data", "X") #Carpeta de la letra X
folder_Y= os.path.join(current_directory, "Detector", "data", "Y") #Carpeta de la letra Y
folder_Z= os.path.join(current_directory, "Detector", "data", "Z") #Carpeta de la letra Z

counter=0 #Contador de imagenes

while True : #Bucle infinito
    success, img = cap.read() #Lee la imagen de la camara
    hands, img = detector.findHands(img) #Detecta las manos en la imagen
    if hands: #Si hay manos
        hand = hands[0] #Toma la primera mano
        x, y, w, h = hand['bbox'] #Obtiene el rectangulo que encierra la mano

        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255 #Crea una imagen blanca
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset] #Recorta la imagen

        imgCropShape = imgCrop.shape #Obtiene el tamaño de la imagen recortada

        

        aspectRatio = h/w #Calcula la relacion de aspecto
        #print(aspectRatio)

        if aspectRatio > 1 : #Si la relacion de aspecto es mayor a 1, la imagen es mas alta que ancha
            k = imgSize / h  #Calcula el factor de escala
            wCal = math.ceil(k * w) #Calcula el ancho de la imagen escalada
            imgResize = cv2.resize(imgCrop,(wCal,imgSize)) #Escalada la imagen
            imgResizeShape = imgResize.shape #Obtiene el tamaño de la imagen escalada
            wGap = math.ceil((imgSize - wCal)/2) #Calcula el margen de la imagen escalada
            imgWhite[:, wGap:wGap + wCal] = imgResize #Pega la imagen escalada en la imagen blanca
        else: #Si la relacion de aspecto es menor a 1, la imagen es mas ancha que alta
            k = imgSize / w #Calcula el factor de escala
            hCal = math.ceil(k * h) #Calcula el alto de la imagen escalada
            imgResize = cv2.resize(imgCrop,(imgSize, hCal)) #Escalada la imagen
            imgResizeShape = imgResize.shape #Obtiene el tamaño de la imagen escalada
            hGap = math.ceil((imgSize - hCal)/2) #Calcula el margen de la imagen escalada
            imgWhite[hGap:hGap + hCal,:] = imgResize #Pega la imagen escalada en la imagen blanca



        cv2.imshow("ImageCrop",imgCrop) #Muestra la imagen recortada
        cv2.imshow("ImageWhite",imgWhite) #Muestra la imagen blanca


    cv2.imshow("Image",img) #Muestra la imagen de la camara
    key = cv2.waitKey(1) #Espera 1 milisegundo por una tecla
    if key == ord("a"): #Si se presiona la tecla a
        counter += 1 #Aumenta el contador
        cv2.imwrite(f'{folder_A}/Image_{time.time()}.jpg',imgWhite) #Guarda la imagen blanca en la carpeta de la letra a con el nombre Image_{time.time()}.jpg
    
    if key == ord("b"): #Si se presiona la tecla b
        counter += 1 #Aumenta el contador
        cv2.imwrite(f'{folder_B}/Image_{time.time()}.jpg',imgWhite) #Guarda la imagen blanca en la carpeta de la letra b con el nombre Image_{time.time()}.jpg

    if key == ord("c"): #Si se presiona la tecla c
        counter += 1 #Aumenta el contador
        cv2.imwrite(f'{folder_C}/Image_{time.time()}.jpg',imgWhite) #Guarda la imagen blanca en la carpeta de la letra C con el nombre Image_{time.time()}.jpg


    if key == ord("d"): #Si se presiona la tecla d
        counter += 1 #Aumenta el contador
        cv2.imwrite(f'{folder_D}/Image_{time.time()}.jpg',imgWhite) #Guarda la imagen blanca en la carpeta de la letra D con el nombre Image_{time.time()}.jpg

    if key == ord("e"): #Si se presiona la tecla e
        counter += 1 #Aumenta el contador
        cv2.imwrite(f'{folder_E}/Image_{time.time()}.jpg',imgWhite) #Guarda la imagen blanca en la carpeta de la letra E con el nombre Image_{time.time()}.jpg

    if key == ord("f"): #Si se presiona la tecla f
        counter += 1 #Aumenta el contador
        cv2.imwrite(f'{folder_F}/Image_{time.time()}.jpg',imgWhite)     #Guarda la imagen blanca en la carpeta de la letra F con el nombre Image_{time.time()}.jpg

    if key == ord("g"): #Si se presiona la tecla g
        counter += 1 #Aumenta el contador
        cv2.imwrite(f'{folder_G}/Image_{time.time()}.jpg',imgWhite) #Guarda la imagen blanca en la carpeta de la letra G con el nombre Image_{time.time()}.jpg
    
    if key == ord("h"): #Si se presiona la tecla h
        counter += 1 #Aumenta el contador
        cv2.imwrite(f'{folder_H}/Image_{time.time()}.jpg',imgWhite)     #Guarda la imagen blanca en la carpeta de la letra H con el nombre Image_{time.time()}.jpg
    
    if key == ord("i"): #Si se presiona la tecla i
        counter += 1 #Aumenta el contador
        cv2.imwrite(f'{folder_I}/Image_{time.time()}.jpg',imgWhite)    #Guarda la imagen blanca en la carpeta de la letra I con el nombre Image_{time.time()}.jpg
    
    if key == ord("j"): #Si se presiona la tecla j
        counter += 1 #Aumenta el contador
        cv2.imwrite(f'{folder_J}/Image_{time.time()}.jpg',imgWhite)    #Guarda la imagen blanca en la carpeta de la letra J con el nombre Image_{time.time()}.jpg
    
    if key == ord("k"): #Si se presiona la tecla k
        counter += 1 #Aumenta el contador
        cv2.imwrite(f'{folder_K}/Image_{time.time()}.jpg',imgWhite)     #Guarda la imagen blanca en la carpeta de la letra K con el nombre Image_{time.time()}.jpg
    
    if key == ord("l"): #Si se presiona la tecla l
        counter += 1 #Aumenta el contador
        cv2.imwrite(f'{folder_L}/Image_{time.time()}.jpg',imgWhite)    #Guarda la imagen blanca en la carpeta de la letra L con el nombre Image_{time.time()}.jpg
    
    if key == ord("m"): #Si se presiona la tecla m
        counter += 1 #Aumenta el contador
        cv2.imwrite(f'{folder_M}/Image_{time.time()}.jpg',imgWhite)   #Guarda la imagen blanca en la carpeta de la letra M con el nombre Image_{time.time()}.jpg
    
    if key == ord("n"): #Si se presiona la tecla n
        counter += 1 #Aumenta el contador 
        cv2.imwrite(f'{folder_N}/Image_{time.time()}.jpg',imgWhite)   #Guarda la imagen blanca en la carpeta de la letra N con el nombre Image_{time.time()}.jpg
    
    if key == ord("o"): #Si se presiona la tecla o
        counter += 1 #Aumenta el contador
        cv2.imwrite(f'{folder_O}/Image_{time.time()}.jpg',imgWhite)     #Guarda la imagen blanca en la carpeta de la letra O con el nombre Image_{time.time()}.jpg
    
    if key == ord("p"): #Si se presiona la tecla p
        counter += 1    #Aumenta el contador
        cv2.imwrite(f'{folder_P}/Image_{time.time()}.jpg',imgWhite)    #Guarda la imagen blanca en la carpeta de la letra P con el nombre Image_{time.time()}.jpg
    
    if key == ord("q"):     #Si se presiona la tecla q
        counter += 1       #Aumenta el contador
        cv2.imwrite(f'{folder_Q}/Image_{time.time()}.jpg',imgWhite)   #Guarda la imagen blanca en la carpeta de la letra Q con el nombre Image_{time.time()}.jpg
    
    if key == ord("r"):    #Si se presiona la tecla r
        counter += 1      #Aumenta el contador
        cv2.imwrite(f'{folder_R}/Image_{time.time()}.jpg',imgWhite)     #Guarda la imagen blanca en la carpeta de la letra R con el nombre Image_{time.time()}.jpg
     
    if key == ord("s"):   #Si se presiona la tecla s
        counter += 1    #Aumenta el contador
        cv2.imwrite(f'{folder_S}/Image_{time.time()}.jpg',imgWhite)   #Guarda la imagen blanca en la carpeta de la letra S con el nombre Image_{time.time()}.jpg

    if key == ord("t"): #Si se presiona la tecla t
        counter += 1 #Aumenta el contador
        cv2.imwrite(f'{folder_T}/Image_{time.time()}.jpg',imgWhite)    #Guarda la imagen blanca en la carpeta de la letra T con el nombre Image_{time.time()}.jpg
    
    if key == ord("u"): #Si se presiona la tecla u
        counter += 1 #Aumenta el contador
        cv2.imwrite(f'{folder_U}/Image_{time.time()}.jpg',imgWhite)   #Guarda la imagen blanca en la carpeta de la letra U con el nombre Image_{time.time()}.jpg
      
    if key == ord("v"): #Si se presiona la tecla v
        counter += 1 #Aumenta el contador
        cv2.imwrite(f'{folder_V}/Image_{time.time()}.jpg',imgWhite)   #Guarda la imagen blanca en la carpeta de la letra V con el nombre Image_{time.time()}.jpg
    
    if key == ord("w"): #Si se presiona la tecla w
        counter += 1 #Aumenta el contador
        cv2.imwrite(f'{folder_W}/Image_{time.time()}.jpg',imgWhite)  #Guarda la imagen blanca en la carpeta de la letra W con el nombre Image_{time.time()}.jpg
    
    if key == ord("x"): #Si se presiona la tecla x
        counter += 1 #Aumenta el contador
        cv2.imwrite(f'{folder_X}/Image_{time.time()}.jpg',imgWhite)  #Guarda la imagen blanca en la carpeta de la letra X con el nombre Image_{time.time()}.jpg
    
    if key == ord("y"): #Si se presiona la tecla y
        counter += 1 #Aumenta el contador
        cv2.imwrite(f'{folder_Y}/Image_{time.time()}.jpg',imgWhite) #Guarda la imagen blanca en la carpeta de la letra Y con el nombre Image_{time.time()}.jpg
    
    if key == ord("z"): #Si se presiona la tecla z
        counter += 1 #Aumenta el contador
        cv2.imwrite(f'{folder_Z}/Image_{time.time()}.jpg',imgWhite) #Guarda la imagen blanca en la carpeta de la letra Z con el nombre Image_{time.time()}.jpg
        

    