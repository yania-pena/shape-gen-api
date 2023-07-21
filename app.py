import cv2
import numpy as np

# Cargar la imagen del cuerpo humano
imagen = cv2.imread('prueba.png')

# Convertir la imagen a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Aplicar umbral adaptativo para obtener una imagen binaria
umbral = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)

# Encontrar contornos en la imagen binaria
contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Crear una imagen en blanco del mismo tamaño que la original
silueta = np.zeros_like(imagen)

# Dibujar los contornos en la imagen en blanco
cv2.drawContours(silueta, contornos, -1, (255, 255, 255), thickness=cv2.FILLED)


silueta_invertida = cv2.bitwise_not(silueta)


cv2.imwrite('silueta_resultante.jpg', silueta_invertida)


# Mostrar la silueta resultante
cv2.imshow('Silueta', silueta_invertida)
cv2.waitKey(0)
cv2.destroyAllWindows()
