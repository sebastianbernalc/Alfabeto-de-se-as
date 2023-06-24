#--------------------------------------------------------------------------
#-----------------------------Alfabeto De Señas ---------------------------
#--------------------------------------------------------------------------
#----------------------------Coceptos básicos de PDI-----------------------
#--------------------------------------------------------------------------
#-----------------------------Sebastian Bernal Cuaspa----------------------
#--------------------------sebastian.bernalc@udea.edu.co-------------------
#--------------------------------------------------------------------------
#---------------------------Kevin David Martinez Zapata--------------------   
#---------------------------kevin.martinez1@udea.edu.co--------------------
#--------------------------------------------------------------------------
#----------------------------Universidad De Antioquia----------------------
#-----------------------------Ingenieria Electronica-----------------------
#------------------------Procesamiento Digital De Imagenes I---------------


#--------------------------------------------------------------------------
#--1. Inicializo el sistema -----------------------------------------------
#--------------------------------------------------------------------------

import tkinter as tk
from PIL import Image, ImageTk
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
from cvzone.ClassificationModule import Classifier
import os


def mostrar_pantalla_señas_a_texto():
#--------------------------------------------------------------------------
#--2. Abro directorio de modelo enseñado con las imagenes -----------------
#--------------------------------------------------------------------------
    current_directory = os.getcwd()                                                         # obtiene la ruta de la carpeta actual
    model_folder = os.path.join(current_directory, "Detector", "modelo", "keras_model.h5")   #Carpeta del modelo entrenado
    labels_folder = os.path.join(current_directory, "Detector", "modelo", "labels.txt")      #Carpeta de los labels (Letras entrenadas)

    labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]                            #Labels para mostrar letra en pantalla
    cap = cv2.VideoCapture(0)                           #Captura de camara
    detector = HandDetector(maxHands = 1)               #Detector de manos activas
    classifier = Classifier(model_folder,labels_folder) #Modelo de aprendizaje automatico entrenado

    #Parametros recuadro mapeo de mano
    offset = 15
    imgSize = 300

    counter=0
#--------------------------------------------------------------------------
#--3. Loop principal-------------------------------------------------------
#--------------------------------------------------------------------------
    while True :
        #--------------------------------------------------------------------------
        #--4. Deteccion de manos por medio de camara-------------------------------
        #--------------------------------------------------------------------------
        success, img = cap.read()            #Lectura de camara y almacenamiento de variables
        imgOutuput = img.copy()              #Copia del cuadro de video leído para poder dibujar visualizaciones y resultados de detección en ella sin modificar el cuadro original.
        hands, img = detector.findHands(img) #Detectar las manos en el cuadro de video leído

        #si se detectaron manos en el cuadro de video leído 
        if hands:
            hand = hands[0]                                                     #Se extrae la primera mano de la lista 
            x, y, w, h = hand['bbox']                                           #La región de interés de la mano se define mediante las coordenadas del cuadro delimitador
            imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255                #Se crea una nueva imagen  y se llena con el valor blanco
            #Esta imagen blanca se utilizará más adelante como fondo para mostrar la mano recortada.
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset] #Se extrae el cuadro de la imagen que rodea la mano y se agrega un margen  a su alrededor para que la mano no quede demasiado cerca de los bordes de la imagen.
            imgCropShape = imgCrop.shape                                        #Se calcula la forma (dimensiones) de la imagen de la mano recortada
            aspectRatio = h/w                                                   #Se calcula el aspectRatio de la mano a partir de su altura (h) y ancho (w).

            #Si la mano es más alta que ancha
            if aspectRatio > 1 :
                k = imgSize / h                                         #Se calcula una escala k que se usará para redimensionar la imagen de la mano (imgCrop) para que su altura sea igual a imgSize
                wCal = math.ceil(k * w)                                 #Se calcula la anchura redimensionada wCal utilizando esta escala y el ancho original de la imagen de la mano.
                imgResize = cv2.resize(imgCrop,(wCal,imgSize))          #Redimensionar la imagen de la mano
                imgResizeShape = imgResize.shape                        #Se calcula la forma (dimensiones) de la imagen de la mano redimensionada
                wGap = math.ceil((imgSize - wCal)/2)                    #Se calcula la cantidad de espacio en blanco que se debe agregar a cada lado de la imagen redimensionada para que tenga el mismo ancho que imgSize
                imgWhite[:, wGap:wGap + wCal] = imgResize               #Se actualiza la imagen en blanco imgWhite para agregar la imagen redimensionada de la mano (imgResize) en el centro de la imagen blanca
                prediction,index = classifier.getPrediction(imgWhite)   #Se utiliza el clasificador de gestos de mano (classifier) para predecir el gesto de la mano en la imagen blanca redimensionada.
                print(prediction,index)                                 #Muestra de datos
                
            #Si la mano es más ancha que alta listo
            else:
                k = imgSize / w                                         #Se calcula una escala k que se usará para redimensionar la imagen de la mano (imgCrop) para que su altura sea igual a imgSize
                hCal = math.ceil(k * h)                                 #Se calcula la anchura redimensionada wCal utilizando esta escala y el ancho original de la imagen de la mano.
                imgResize = cv2.resize(imgCrop,(imgSize, hCal))         #Redimensionar la imagen de la mano
                imgResizeShape = imgResize.shape                        #Se calcula la forma (dimensiones) de la imagen de la mano redimensionada
                hGap = math.ceil((imgSize - hCal)/2)                    #Se calcula la cantidad de espacio en blanco que se debe agregar a cada lado de la imagen redimensionada para que tenga el mismo ancho que imgSize
                imgWhite[hGap:hGap + hCal,:] = imgResize                #Se actualiza la imagen en blanco imgWhite para agregar la imagen redimensionada de la mano (imgResize) en el centro de la imagen blanca
                prediction,index = classifier.getPrediction(imgWhite)   #Se utiliza el clasificador de gestos de mano (classifier) para predecir el gesto de la mano en la imagen blanca redimensionada.
            #--------------------------------------------------------------------------
            #--5. Predictor de mano captada por camara con respecto a modelo enseñado--
            #--------------------------------------------------------------------------
            cv2.putText(imgOutuput, labels[index], (x, y-20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)

            cv2.imshow("ImageCrop",imgCrop)
            cv2.imshow("ImageWhite",imgWhite)


        cv2.imshow("Image",imgOutuput)
        
        key = cv2.waitKey(1)
        if key == 27:  # Tecla 'Esc' para salir del bucle
            break

    cap.release()
    cv2.destroyAllWindows()

#--------------------------------------------------------------------------
#--6. Pantalla de texto a señas--------------------------------------------
#--------------------------------------------------------------------------

def mostrar_pantalla_texto_a_señas():
    # Código para mostrar la pantalla "De Texto a Señas"
    def enviar_texto():
        texto_ingresado = cuadro_texto.get("1.0", "end-1c").lower()  # Obtener el texto ingresado en el cuadro de texto en minúsculas

        if texto_ingresado == "a":
            # Mostrar la imagen "hola.jpg" en la parte inferior
            imagen_resultante = Image.open("Detector/images/A/Captura.jpg")  # Reemplaza "ruta/hola.jpg" por la ruta absoluta de tu imagen
            imagen_resultante = imagen_resultante.resize((500, 300))  # Ajustar el tamaño de la imagen según tus necesidades
            imagen_resultante = ImageTk.PhotoImage(imagen_resultante)
            etiqueta_imagen.configure(image=imagen_resultante)
            etiqueta_imagen.image = imagen_resultante

        elif texto_ingresado == "b":
            # Mostrar la imagen "hola.jpg" en la parte inferior
            imagen_resultante = Image.open("Detector/images/B/Captura.jpg")  # Reemplaza "ruta/hola.jpg" por la ruta absoluta de tu imagen
            imagen_resultante = imagen_resultante.resize((500, 300))  # Ajustar el tamaño de la imagen según tus necesidades
            imagen_resultante = ImageTk.PhotoImage(imagen_resultante)
            etiqueta_imagen.configure(image=imagen_resultante)
            etiqueta_imagen.image = imagen_resultante
        elif texto_ingresado == "c":
            # Mostrar la imagen "hola.jpg" en la parte inferior
            imagen_resultante = Image.open("Detector/images/C/Captura.jpg")  # Reemplaza "ruta/hola.jpg" por la ruta absoluta de tu imagen
            imagen_resultante = imagen_resultante.resize((500, 300))  # Ajustar el tamaño de la imagen según tus necesidades
            imagen_resultante = ImageTk.PhotoImage(imagen_resultante)
            etiqueta_imagen.configure(image=imagen_resultante)
            etiqueta_imagen.image = imagen_resultante
        elif texto_ingresado == "d":
            # Mostrar la imagen "hola.jpg" en la parte inferior
            imagen_resultante = Image.open("Detector/images/D/Captura.jpg")  # Reemplaza "ruta/hola.jpg" por la ruta absoluta de tu imagen
            imagen_resultante = imagen_resultante.resize((500, 300))  # Ajustar el tamaño de la imagen según tus necesidades
            imagen_resultante = ImageTk.PhotoImage(imagen_resultante)
            etiqueta_imagen.configure(image=imagen_resultante)
            etiqueta_imagen.image = imagen_resultante
        elif texto_ingresado == "e":
            # Mostrar la imagen "hola.jpg" en la parte inferior
            imagen_resultante = Image.open("Detector/images/E/Captura.jpg")  # Reemplaza "ruta/hola.jpg" por la ruta absoluta de tu imagen
            imagen_resultante = imagen_resultante.resize((500, 300))  # Ajustar el tamaño de la imagen según tus necesidades
            imagen_resultante = ImageTk.PhotoImage(imagen_resultante)
            etiqueta_imagen.configure(image=imagen_resultante)
            etiqueta_imagen.image = imagen_resultante
        elif texto_ingresado == "f":
            # Mostrar la imagen "hola.jpg" en la parte inferior
            imagen_resultante = Image.open("Detector/images/F/Captura.jpg")  # Reemplaza "ruta/hola.jpg" por la ruta absoluta de tu imagen
            imagen_resultante = imagen_resultante.resize((500, 300))  # Ajustar el tamaño de la imagen según tus necesidades
            imagen_resultante = ImageTk.PhotoImage(imagen_resultante)
            etiqueta_imagen.configure(image=imagen_resultante)
            etiqueta_imagen.image = imagen_resultante
        elif texto_ingresado == "g":
            # Mostrar la imagen "hola.jpg" en la parte inferior
            imagen_resultante = Image.open("Detector/images/G/Captura.jpg")  # Reemplaza "ruta/hola.jpg" por la ruta absoluta de tu imagen
            imagen_resultante = imagen_resultante.resize((500, 300))  # Ajustar el tamaño de la imagen según tus necesidades
            imagen_resultante = ImageTk.PhotoImage(imagen_resultante)
            etiqueta_imagen.configure(image=imagen_resultante)
            etiqueta_imagen.image = imagen_resultante
        elif texto_ingresado == "h":
            # Mostrar la imagen "hola.jpg" en la parte inferior
            imagen_resultante = Image.open("Detector/images/H/Captura.jpg")  # Reemplaza "ruta/hola.jpg" por la ruta absoluta de tu imagen
            imagen_resultante = imagen_resultante.resize((500, 300))  # Ajustar el tamaño de la imagen según tus necesidades
            imagen_resultante = ImageTk.PhotoImage(imagen_resultante)
            etiqueta_imagen.configure(image=imagen_resultante)
            etiqueta_imagen.image = imagen_resultante
        elif texto_ingresado == "i":
            # Mostrar la imagen "hola.jpg" en la parte inferior
            imagen_resultante = Image.open("Detector/images/I/Captura.jpg")  # Reemplaza "ruta/hola.jpg" por la ruta absoluta de tu imagen
            imagen_resultante = imagen_resultante.resize((500, 300))  # Ajustar el tamaño de la imagen según tus necesidades
            imagen_resultante = ImageTk.PhotoImage(imagen_resultante)
            etiqueta_imagen.configure(image=imagen_resultante)
            etiqueta_imagen.image = imagen_resultante
        elif texto_ingresado == "j":
            # Mostrar la imagen "hola.jpg" en la parte inferior
            imagen_resultante = Image.open("Detector/images/J/Captura.jpg")  # Reemplaza "ruta/hola.jpg" por la ruta absoluta de tu imagen
            imagen_resultante = imagen_resultante.resize((500, 300))  # Ajustar el tamaño de la imagen según tus necesidades
            imagen_resultante = ImageTk.PhotoImage(imagen_resultante)
            etiqueta_imagen.configure(image=imagen_resultante)
            etiqueta_imagen.image = imagen_resultante
        elif texto_ingresado == "k":
            # Mostrar la imagen "hola.jpg" en la parte inferior
            imagen_resultante = Image.open("Detector/images/K/Captura.jpg")  # Reemplaza "ruta/hola.jpg" por la ruta absoluta de tu imagen
            imagen_resultante = imagen_resultante.resize((500, 300))  # Ajustar el tamaño de la imagen según tus necesidades
            imagen_resultante = ImageTk.PhotoImage(imagen_resultante)
            etiqueta_imagen.configure(image=imagen_resultante)
            etiqueta_imagen.image = imagen_resultante
        elif texto_ingresado == "l":
            # Mostrar la imagen "hola.jpg" en la parte inferior
            imagen_resultante = Image.open("Detector/images/L/Captura.jpg")  # Reemplaza "ruta/hola.jpg" por la ruta absoluta de tu imagen
            imagen_resultante = imagen_resultante.resize((500, 300))  # Ajustar el tamaño de la imagen según tus necesidades
            imagen_resultante = ImageTk.PhotoImage(imagen_resultante)
            etiqueta_imagen.configure(image=imagen_resultante)
            etiqueta_imagen.image = imagen_resultante
        elif texto_ingresado == "m":
            # Mostrar la imagen "hola.jpg" en la parte inferior
            imagen_resultante = Image.open("Detector/images/M/Captura.jpg")  # Reemplaza "ruta/hola.jpg" por la ruta absoluta de tu imagen
            imagen_resultante = imagen_resultante.resize((500, 300))  # Ajustar el tamaño de la imagen según tus necesidades
            imagen_resultante = ImageTk.PhotoImage(imagen_resultante)
            etiqueta_imagen.configure(image=imagen_resultante)
            etiqueta_imagen.image = imagen_resultante
        elif texto_ingresado == "n":
            # Mostrar la imagen "hola.jpg" en la parte inferior
            imagen_resultante = Image.open("Detector/images/N/Captura.jpg")  # Reemplaza "ruta/hola.jpg" por la ruta absoluta de tu imagen
            imagen_resultante = imagen_resultante.resize((500, 300))  # Ajustar el tamaño de la imagen según tus necesidades
            imagen_resultante = ImageTk.PhotoImage(imagen_resultante)
            etiqueta_imagen.configure(image=imagen_resultante)
            etiqueta_imagen.image = imagen_resultante
        elif texto_ingresado == "o":
            # Mostrar la imagen "hola.jpg" en la parte inferior
            imagen_resultante = Image.open("Detector/images/O/Captura.jpg")  # Reemplaza "ruta/hola.jpg" por la ruta absoluta de tu imagen
            imagen_resultante = imagen_resultante.resize((500, 300))  # Ajustar el tamaño de la imagen según tus necesidades
            imagen_resultante = ImageTk.PhotoImage(imagen_resultante)
            etiqueta_imagen.configure(image=imagen_resultante)
            etiqueta_imagen.image = imagen_resultante
        elif texto_ingresado == "p":
            # Mostrar la imagen "hola.jpg" en la parte inferior
            imagen_resultante = Image.open("Detector/images/P/Captura.jpg")  # Reemplaza "ruta/hola.jpg" por la ruta absoluta de tu imagen
            imagen_resultante = imagen_resultante.resize((500, 300))  # Ajustar el tamaño de la imagen según tus necesidades
            imagen_resultante = ImageTk.PhotoImage(imagen_resultante)
            etiqueta_imagen.configure(image=imagen_resultante)
            etiqueta_imagen.image = imagen_resultante
        elif texto_ingresado == "q":
            # Mostrar la imagen "hola.jpg" en la parte inferior
            imagen_resultante = Image.open("Detector/images/Q/Captura.jpg")  # Reemplaza "ruta/hola.jpg" por la ruta absoluta de tu imagen
            imagen_resultante = imagen_resultante.resize((500, 300))  # Ajustar el tamaño de la imagen según tus necesidades
            imagen_resultante = ImageTk.PhotoImage(imagen_resultante)
            etiqueta_imagen.configure(image=imagen_resultante)
            etiqueta_imagen.image = imagen_resultante
        elif texto_ingresado == "r":
            # Mostrar la imagen "hola.jpg" en la parte inferior
            imagen_resultante = Image.open("Detector/images/R/Captura.jpg")  # Reemplaza "ruta/hola.jpg" por la ruta absoluta de tu imagen
            imagen_resultante = imagen_resultante.resize((500, 300))  # Ajustar el tamaño de la imagen según tus necesidades
            imagen_resultante = ImageTk.PhotoImage(imagen_resultante)
            etiqueta_imagen.configure(image=imagen_resultante)
            etiqueta_imagen.image = imagen_resultante
        elif texto_ingresado == "s":
            # Mostrar la imagen "hola.jpg" en la parte inferior
            imagen_resultante = Image.open("Detector/images/S/Captura.jpg")  # Reemplaza "ruta/hola.jpg" por la ruta absoluta de tu imagen
            imagen_resultante = imagen_resultante.resize((500, 300))  # Ajustar el tamaño de la imagen según tus necesidades
            imagen_resultante = ImageTk.PhotoImage(imagen_resultante)
            etiqueta_imagen.configure(image=imagen_resultante)
            etiqueta_imagen.image = imagen_resultante
        elif texto_ingresado == "t":
            # Mostrar la imagen "hola.jpg" en la parte inferior
            imagen_resultante = Image.open("Detector/images/T/Captura.jpg")  # Reemplaza "ruta/hola.jpg" por la ruta absoluta de tu imagen
            imagen_resultante = imagen_resultante.resize((500, 300))  # Ajustar el tamaño de la imagen según tus necesidades
            imagen_resultante = ImageTk.PhotoImage(imagen_resultante)
            etiqueta_imagen.configure(image=imagen_resultante)
            etiqueta_imagen.image = imagen_resultante
        elif texto_ingresado == "u":
            # Mostrar la imagen "hola.jpg" en la parte inferior
            imagen_resultante = Image.open("Detector/images/U/Captura.jpg")  # Reemplaza "ruta/hola.jpg" por la ruta absoluta de tu imagen
            imagen_resultante = imagen_resultante.resize((500, 300))  # Ajustar el tamaño de la imagen según tus necesidades
            imagen_resultante = ImageTk.PhotoImage(imagen_resultante)
            etiqueta_imagen.configure(image=imagen_resultante)
            etiqueta_imagen.image = imagen_resultante
        elif texto_ingresado == "v":
            # Mostrar la imagen "hola.jpg" en la parte inferior
            imagen_resultante = Image.open("Detector/images/V/Captura.jpg")  # Reemplaza "ruta/hola.jpg" por la ruta absoluta de tu imagen
            imagen_resultante = imagen_resultante.resize((500, 300))  # Ajustar el tamaño de la imagen según tus necesidades
            imagen_resultante = ImageTk.PhotoImage(imagen_resultante)
            etiqueta_imagen.configure(image=imagen_resultante)
            etiqueta_imagen.image = imagen_resultante
        elif texto_ingresado == "w":
            # Mostrar la imagen "hola.jpg" en la parte inferior
            imagen_resultante = Image.open("Detector/images/W/Captura.jpg")  # Reemplaza "ruta/hola.jpg" por la ruta absoluta de tu imagen
            imagen_resultante = imagen_resultante.resize((500, 300))  # Ajustar el tamaño de la imagen según tus necesidades
            imagen_resultante = ImageTk.PhotoImage(imagen_resultante)
            etiqueta_imagen.configure(image=imagen_resultante)
            etiqueta_imagen.image = imagen_resultante
        elif texto_ingresado == "x":
            # Mostrar la imagen "hola.jpg" en la parte inferior
            imagen_resultante = Image.open("Detector/images/X/Captura.jpg")  # Reemplaza "ruta/hola.jpg" por la ruta absoluta de tu imagen
            imagen_resultante = imagen_resultante.resize((500, 300))  # Ajustar el tamaño de la imagen según tus necesidades
            imagen_resultante = ImageTk.PhotoImage(imagen_resultante)
            etiqueta_imagen.configure(image=imagen_resultante)
            etiqueta_imagen.image = imagen_resultante
        elif texto_ingresado == "y":
            # Mostrar la imagen "hola.jpg" en la parte inferior
            imagen_resultante = Image.open("Detector/images/Y/Captura.jpg")  # Reemplaza "ruta/hola.jpg" por la ruta absoluta de tu imagen
            imagen_resultante = imagen_resultante.resize((500, 300))  # Ajustar el tamaño de la imagen según tus necesidades
            imagen_resultante = ImageTk.PhotoImage(imagen_resultante)
            etiqueta_imagen.configure(image=imagen_resultante)
            etiqueta_imagen.image = imagen_resultante
        elif texto_ingresado == "z":
            # Mostrar la imagen "hola.jpg" en la parte inferior
            imagen_resultante = Image.open("Detector/images/Z/Captura.jpg")  # Reemplaza "ruta/hola.jpg" por la ruta absoluta de tu imagen
            imagen_resultante = imagen_resultante.resize((500, 300))  # Ajustar el tamaño de la imagen según tus necesidades
            imagen_resultante = ImageTk.PhotoImage(imagen_resultante)
            etiqueta_imagen.configure(image=imagen_resultante)
            etiqueta_imagen.image = imagen_resultante
        
        else:
            # Limpiar la imagen resultante si no coincide con la palabra "a"
            etiqueta_imagen.configure(image=None)
            etiqueta_imagen.image = None
#--------------------------------------------------------------------------
#--7. Definicion de botones y etiquetas------------------------------------
#--------------------------------------------------------------------------
    def volver():
        ventana_texto_a_señas.destroy()  # Cerrar la ventana "De Texto a Señas"

    # Crear la ventana "De Texto a Señas"
    ventana_texto_a_señas = tk.Toplevel(ventana)
    ventana_texto_a_señas.title("De Texto a Señas")
    ventana_texto_a_señas.geometry("1280x720")

    # Cuadro de texto para ingresar el texto
    cuadro_texto = tk.Text(ventana_texto_a_señas, height=5, width=30)
    cuadro_texto.pack(pady=10)

    # Botón "Enviar"
    boton_enviar = tk.Button(ventana_texto_a_señas, text="Enviar", font=("Corbel Light", 12), width=10, height=1, command=enviar_texto)
    boton_enviar.pack(pady=5)

    # Etiqueta para mostrar la imagen resultante
    etiqueta_imagen = tk.Label(ventana_texto_a_señas)
    etiqueta_imagen.pack(pady=10)

    # Botón "Volver"
    boton_volver = tk.Button(ventana_texto_a_señas, text="Volver", font=("Corbel Light", 12), width=10, height=1, command=volver)
    boton_volver.pack(pady=5)

# Crear la ventana principal
ventana = tk.Tk()
ventana.title("Alfabeto de Lenguaje de Señas")
ventana.geometry("1280x720")  # Establecer el tamaño de la ventana

# Cargar la imagen
imagen_path = "Detector/udea.png"
imagen = Image.open(imagen_path)
imagen = imagen.resize((500, 150), Image.ANTIALIAS)  # Redimensionar la imagen al tamaño deseado
imagen_tk = ImageTk.PhotoImage(imagen)

# Crear el widget Label para mostrar la imagen
label_imagen = tk.Label(ventana, image=imagen_tk)
label_imagen.pack(pady=5)

# Etiqueta de bienvenida
etiqueta_bienvenida = tk.Label(ventana, text="BIENVENIDOS AL ALFABETO DE LENGUAJE DE SEÑAS\nA CARGO DE SEBASTIAN BERNAL-KEVIN MARTINEZ\nINGENIERIA ELECTRONICA", font=("Corbel Light", 12))
etiqueta_bienvenida.pack(pady=20)


# Botón "De Señas a Texto"
boton_señas_a_texto = tk.Button(ventana, text="DE SEÑAS A TEXTO", font=("Corbel Light", 12), width=20, height=3, command=mostrar_pantalla_señas_a_texto)
boton_señas_a_texto.pack(pady=30)



# Botón "De Texto a Señas"
boton_texto_a_señas = tk.Button(ventana, text="DE TEXTO A SEÑAS", font=("Corbel Light", 12), width=20, height=3, command=mostrar_pantalla_texto_a_señas)
boton_texto_a_señas.pack(pady=30)

# Ejecutar la ventana principal
ventana.mainloop()
