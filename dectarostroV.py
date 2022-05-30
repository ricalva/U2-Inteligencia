from platform import release
from re import S
from sys import maxsize
import cv2
from cv2 import rectangle
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
import numpy as np

DetectaRostro = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#*Dectar sobre Video

cam = cv2.VideoCapture(0)

#*Cambiar a las escala de gris
while True:
    ret,frame = cam.read()
    if ret == True:
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#*VARIABLES Y FUNCIONES PARA DETECCION DE ROSTROS

    caras = DetectaRostro.detectMultiScale(
    gris, 
    scaleFactor = 1.1, 
    minNeighbors = S,
    minSize = (30,30),
    maxSize = (0, 200)
)
#*Dibujar los rostros detectados

    for (x1, y1, x2, y2) in caras:
            cv2.rectangle(frame, (x1, y2), (x1 + x2, y1+y2), (255, 255, 255), 2)

#* Mostrar la imagen
            cv2.imshow('Imagen Dectectada', frame)

            if cv2.waitKey(1) & 0xFF == ord('a'):
                break

    cam.release()
    cv2.destroyAllWindows()

    