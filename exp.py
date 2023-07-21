import cv2

# Cargar la imagen de la silueta (asegúrate de tenerla en la ruta correcta)
ruta_silueta = 'silueta_resultante.jpg'
silueta = cv2.imread(ruta_silueta, cv2.IMREAD_GRAYSCALE)

# Aplicar umbralización para convertir en imagen binaria
_, silueta_binaria = cv2.threshold(silueta, 0, 255, cv2.THRESH_BINARY)

# Encontrar los contornos en la imagen binaria
contornos, _ = cv2.findContours(silueta_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtrar y obtener la cabeza
cabeza = None
for contorno in contornos:
    x, y, w, h = cv2.boundingRect(contorno)
    
    # Filtrar por tamaño y posición para obtener la cabeza
    if h > 0.3 * silueta.shape[0] and w > 0.3 * silueta.shape[1]:
        cabeza = silueta[y:y+h, x:x+w]
        break

# Mostrar la cabeza en una ventana
cv2.imshow('Cabeza', cabeza)
cv2.waitKey(0)
cv2.destroyAllWindows()