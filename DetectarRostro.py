from re import S
from sys import maxsize
import cv2
from cv2 import rectangle
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
import numpy as np

detectarostro = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#*Dectar sobre imagen/Foto

foto = cv2.imread('amigos2.jpg')

#*Cambiar a las escala de gris
gris = cv2.cvtColor(foto, cv2.COLOR_BGR2GRAY)

#*VARIABLES Y FUNCIONES PARA DETECCION DE ROSTROS

caras = detectarostro.detectMultiScale(
    gris,
    scaleFactor = 1.2,
    minNeighbors = 5,
    minSize = (30,30),
    maxSize = (0,200)
)
#*Dibujar los rostros detectados

for (x1, y1, x2, y2) in caras:
    cv2.rectangle(foto, (x1, y1), (x1 + x2, y1+y2), (255, 255, 255), 2)

#* Mostrar la imagen
cv2.imshow('Imagen Dectectada', foto)
cv2.waitKey(0)
cv2.destroyAllWindows()