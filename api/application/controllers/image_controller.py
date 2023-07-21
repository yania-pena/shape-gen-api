
from flask.json import jsonify
from api.application.utils import *
import cv2
import numpy as np
import requests


def new_method(file, months=5):
    original = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite('original.jpg', original)

    results = list()

    results.append({
        "name": "original.jpg",
        "url": upload("original.jpg")
    })
    #cv2.imshow("original", original) 
    # Convertimos a escala de grises
    gris = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    
    # Aplicar suavizado Gaussiano
    gauss = cv2.GaussianBlur(gris, (5,5), 0)
    
    #cv2.imshow("suavizado", gauss)
    
    # Detectamos los bordes con Canny
    canny = cv2.Canny(gauss, 50, 150)
    
    #cv2.imshow("canny", canny)



    # Buscamos los contornos
    (contornos,_) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Mostramos el número de monedas por consola
    #print("He encontrado {} objetos".format(len(contornos)))
    
    fondo = np.ones_like(original) * 255
    cv2.drawContours(fondo, contornos, -1, (0, 0, 0), 2)
    #cv2.imshow("contornos", fondo)
    
    cv2.imwrite("contorno.jpg", fondo)
    results.append({
        "name": 'contorno.jpg',
        "url": upload(f'contorno.jpg')
    })

    contorno_compuesto = np.concatenate(contornos)

    for i in range(months):
        reduction_factor = 1-(0.1*i)  # Ajusta este valor según tus necesidades

        # Calcula el punto medio del contorno en el eje x
        x_mean = np.mean(contorno_compuesto[:, 0, 0])

        # Calcula el desplazamiento basado en el punto medio y el factor de reducción
        shift = (1 - reduction_factor) * x_mean

        # Aplica la reducción horizontal al contorno
        reduced_contour = contorno_compuesto.copy()  # Copia el contorno original
        reduced_contour[:, 0, 0] = (reduced_contour[:, 0, 0] - shift) * reduction_factor + shift

        
        image_2 = np.ones_like(original) * 255  # Crea una imagen en blanco para dibujar
        cv2.drawContours(image_2, [reduced_contour], -1, (0, 0, 0), 2)  


        gris = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    
        # Aplicar suavizado Gaussiano
        gauss = cv2.GaussianBlur(gris, (5,5), 0)
                
        filename = f'imagen{i}.jpg'
        
        cv2.imwrite(filename, gauss)
        results.append({
            "name": filename,
            "url": upload(filename)
        })

        """
        filename = f'imagen{i}.jpg'
        #cv2.imshow(filename, image_2)
        cv2.imwrite(filename, image_2)
        
        results.append({
            "name": filename,
            "url": upload(filename)
        })
        """

    return jsonify({"status": True, "url_images": results })
    

def generate_shape(file, months=5):
    results = list()
    
    imagen = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    cv2.imwrite('original.jpg', imagen)
    results.append({
        "name": "original.jpg",
        "url": upload("original.jpg")
    })

    
    """
    # Cargar la imagen del cuerpo humano
    #imagen = cv2.imread('prueba.png')

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


    cv2.imwrite('silueta_resultante_3.jpg', silueta_invertida)
    """
    # Convertir la imagen de BGR a HSV
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    # Definir el rango de color para el blanco
    rango_blanco_bajo = np.array([0, 0, 200])
    rango_blanco_alto = np.array([180, 30, 255])

    # Crear una máscara para el color blanco
    mascara_blanco = cv2.inRange(hsv, rango_blanco_bajo, rango_blanco_alto)

    # Invertir la máscara para excluir el color blanco
    mascara_excluida = cv2.bitwise_not(mascara_blanco)

    # Aplicar operaciones morfológicas para mejorar la máscara excluida
    kernel = np.ones((5, 5), np.uint8)
    mascara_excluida = cv2.morphologyEx(mascara_excluida, cv2.MORPH_OPEN, kernel)

    # Encontrar los contornos en la máscara excluida
    contornos, _ = cv2.findContours(mascara_excluida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # Crear una imagen en blanco del mismo tamaño que la original
    imagen_contorno = np.zeros_like(imagen)

    # Dibujar los contornos en la imagen de contorno
    cv2.drawContours(imagen_contorno, contornos, -1, (0, 255, 0), 2)

    cv2.imwrite('imagen_contorno.jpg', imagen_contorno)



    contour = contornos[0]
    # Dibuja el contorno original y el contorno reducido en una imagen
    image = np.ones((500, 500, 3), dtype=np.uint8)  * 255  # Crea una imagen en blanco para dibujar
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)  # Dibuja el contorno original en verde

    # Muestra la imagen con los contornos dibujados
    # cv2.imshow('Contornos', image)
    cv2.imwrite('contorno.jpg', image)    
    results.append({
        "name": 'contorno.jpg',
        "url": upload(f'contorno.jpg')
    })


    for i in range(months):
        reduction_factor = 1-(0.1*i)  # Ajusta este valor según tus necesidades

        # Calcula el punto medio del contorno en el eje x
        x_mean = np.mean(contour[:, 0, 0])

        # Calcula el desplazamiento basado en el punto medio y el factor de reducción
        shift = (1 - reduction_factor) * x_mean

        # Aplica la reducción horizontal al contorno
        reduced_contour = contour.copy()  # Copia el contorno original
        reduced_contour[:, 0, 0] = (reduced_contour[:, 0, 0] - shift) * reduction_factor + shift

        
        image_2 = np.ones((500, 500, 3), dtype=np.uint8) * 255  # Crea una imagen en blanco para dibujar
        cv2.drawContours(image_2, [reduced_contour], -1, (0, 0, 255), 2)  
        
        filename = f'imagen{i}.jpg'
        cv2.imwrite(filename, image_2)
        
        results.append({
            "name": filename,
            "url": upload(filename)
        })



    
    # Dibuja el contorno original y el contorno reducido en una imagen
    #image = np.zeros((500, 500, 3), dtype=np.uint8)  # Crea una imagen en blanco para dibujar
    #cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)  # Dibuja el contorno original en verde

    # Muestra la imagen con los contornos dibujados
    #cv2.imshow('Contornos', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    #Mostrar solo el contorno
    # cv2.imshow('Contorno', imagen_contorno)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    #cv2.imwrite('silueta_resultante_3.jpg', imagen)
    
    #silueta_clean = remove_background('silueta_resultante_3.jpg')
    #return upload(silueta_clean)
    #return upload('silueta_resultante_3.jpg')
    #return crop('imagen_contorno.jpg')
    # Mostrar la silueta resultante
    # cv2.imshow('Silueta', silueta_invertida)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #return jsonify({"status": True, "url_images":  })
    return jsonify({"status": True, "url_images": results })



def get_contours(cropped_image):
    #a
    # Convertir la imagen de BGR a HSV
    hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

    # Definir el rango de color para el blanco
    rango_blanco_bajo = np.array([0, 0, 200])
    rango_blanco_alto = np.array([180, 30, 255])

    # Crear una máscara para el color blanco
    mascara_blanco = cv2.inRange(hsv, rango_blanco_bajo, rango_blanco_alto)

    # Invertir la máscara para excluir el color blanco
    mascara_excluida = cv2.bitwise_not(mascara_blanco)

    # Aplicar operaciones morfológicas para mejorar la máscara excluida
    kernel = np.ones((5, 5), np.uint8)
    mascara_excluida = cv2.morphologyEx(mascara_excluida, cv2.MORPH_OPEN, kernel)

    # Encontrar los contornos en la máscara excluida
    contornos, _ = cv2.findContours(mascara_excluida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # Crear una imagen en blanco del mismo tamaño que la original
    imagen_contorno = np.zeros_like(cropped_image)

    # Dibujar los contornos en la imagen de contorno
    # cv2.drawContours(imagen_contorno, contornos, -1, (0, 255, 0), 2)
    # cv2.imshow('imagen_contorno', imagen_contorno)
    #b

    #c
    contour = contornos[0]
    return contour


def new_get_con(cropped_image, ancho_deseado, filename):
    #cv2.imshow("original", original) 
    # Convertimos a escala de grises
    gris = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar suavizado Gaussiano
    gauss = cv2.GaussianBlur(gris, (5,5), 0)
    
    #cv2.imshow("suavizado", gauss)
    
    # Detectamos los bordes con Canny
    canny = cv2.Canny(gauss, 50, 150)
    
    #cv2.imshow("canny", canny)

    # Buscamos los contornos
    (contornos,_) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crear una imagen en blanco del mismo tamaño que la imagen original
    imagen_contornos = np.zeros_like(cropped_image)

    # Dibujar los contornos en la imagen en blanco
    cv2.drawContours(imagen_contornos, contornos, -1, (255, 255, 255), thickness=cv2.FILLED)

    imagen_contornos = cv2.bitwise_not(imagen_contornos)


    # Mostrar la imagen con los contornos unidos
    # cv2.imshow('Contornos unidos', imagen_contornos)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #contorno_compuesto = np.concatenate(contornos)


    # Definir el factor de escala horizontal (ajustar según el aumento o reducción de peso deseado)
    # factor_escala_horizontal = 2.4

    # Obtener las dimensiones de la imagen original
    alto, ancho = imagen_contornos.shape[:2]

    # Calcular el nuevo ancho de la imagen escalada
    base_value = 300
    nuevo_ancho = ancho_deseado*4 if ancho_deseado>1 else ancho

    # Escalar la imagen horizontalmente
    imagen_escalada = cv2.resize(imagen_contornos, (nuevo_ancho, alto))

    # Mostrar la imagen escalada
    # cv2.imshow('Imagen escalada', imagen_escalada)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    cv2.imwrite(filename, imagen_escalada)
    return imagen_escalada
    #return upload(filename)
    #return imagen_escalada


def transform_countour(value, contour, cropped_image, filename):
    clean_value = int(value)/-40 if int(value) < 80 else int(value)/40
    reduction_factor = 1+(0.1*clean_value)  # Ajusta este valor según tus necesidades

    # Calcula el punto medio del contorno en el eje x
    x_mean = np.mean(contour[:, 0, 0])

    # Calcula el desplazamiento basado en el punto medio y el factor de reducción
    shift = (1 - reduction_factor) * x_mean

    # Aplica la reducción horizontal al contorno
    reduced_contour = contour.copy()  # Copia el contorno original
    reduced_contour[:, 0, 0] = (reduced_contour[:, 0, 0] - shift) * reduction_factor + shift
    
    height, width, channels = cropped_image.shape 
    image_2 = np.ones((height, width+200, channels), dtype=np.uint8) * 255  # Crea una imagen en blanco para dibujar
    cv2.drawContours(image_2, [reduced_contour], -1, (0, 0, 255), 2)  
    # cv2.imshow('file', image_2)
    #d
    
    # Muestra la imagen original y la imagen cortada
    # cv2.imshow('Imagen original', image)
    # cv2.imshow('Imagen cortada', cropped_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imwrite(filename, image_2)
    return upload(filename)
    



def rellenar_altura(imagen, nueva_altura):
    altura_actual = imagen.shape[0]
    ancho = imagen.shape[1]
    
    # Calcular la cantidad de píxeles en blanco que se necesitan para el relleno
    pixels_en_blanco = np.full((nueva_altura - altura_actual, ancho, 3), 255, dtype=np.uint8)
    
    # Concatenar la imagen original con los píxeles en blanco
    imagen_rellena = np.vstack((imagen, pixels_en_blanco))

    return imagen_rellena


def rellenar_ancho(imagen, nuevo_ancho):
    altura = imagen.shape[0]
    ancho_actual = imagen.shape[1]
    
    # Calcular la cantidad de píxeles en blanco que se necesitan para el relleno
    pixels_en_blanco = np.full((altura, nuevo_ancho - ancho_actual, 3), 255, dtype=np.uint8)
    
    # Concatenar la imagen original con los píxeles en blanco
    imagen_rellena = np.hstack((imagen, pixels_en_blanco))

    return imagen_rellena

def tr_image(data):
    img_url = data['image_url']
    cuello_width = data['medida_cuello']
    brazos_width = data['medida_brazos']
    pecho_width = data['medida_pecho']
    cintura_width = data['medida_cintura']
    cadera_width = data['medida_cadera']
    piernas_width = data['medida_piernas']

    print(f"img_url {img_url}")

    # Realiza la descarga de la imagen desde Cloudinary
    response = requests.get(img_url)
    ruta_destino = 'temp.jpg'

    # Verifica si la descarga fue exitosa
    if response.status_code == 200:
        # Guarda la imagen en el servidor
        with open(ruta_destino, 'wb') as file:
            file.write(response.content)
    else:
        print('No se pudo descargar la imagen desde la URL de Cloudinary')
        return jsonify({"status": False, "message": "No se pudo descargar" })

    image = cv2.imread('temp.jpg')   
    
    
    cabeza = image[0:139, :]
    cuello = image[140: 164, :]
    pecho = image[165:289, :]
    cintura = image[290:410, :]
    cadera = image[410: 500, :]
    piernas = image[501: 1000, :]    
    """
    mes = list()
    mes.append(cuello_width)
    mes.append(pecho_width)
    mes.append(cintura_width)
    mes.append(cadera_width)
    mes.append(piernas_width)
    
    bigger = max(mes)
    cuello_width = bigger
    pecho_width = bigger
    cintura_width = bigger
    cadera_width = bigger
    piernas_width = bigger
    """

    cabeza_contorno = new_get_con(cabeza, 1, "cabeza_modif.jpg")
    cuello_contorno = new_get_con(cuello, cuello_width, "cuello_modif.jpg")
    pecho_contorno = new_get_con(pecho, pecho_width, "pecho_modif.jpg")
    cintura_contorno = new_get_con(cintura, cintura_width, "cintura_modif.jpg")
    cadera_contorno = new_get_con(cadera, cadera_width, "cadera_modif.jpg")
    piernas_contorno = new_get_con(piernas, piernas_width, "piernas_modif.jpg")


    # Aplicar el escalado no uniforme a cada parte del cuerpo para mantener las proporciones
    # Aplicar el escalado no uniforme a cada parte del cuerpo para mantener las proporciones
    cabeza_contorno = escalar_proporcional(cabeza_contorno, cabeza, 1, 1)
    cuello_contorno = escalar_proporcional(cuello_contorno, cuello, cuello_width, 1)
    pecho_contorno = escalar_proporcional(pecho_contorno, pecho, pecho_width, 1)
    cintura_contorno = escalar_proporcional(cintura_contorno, cintura, cintura_width, 1)
    cadera_contorno = escalar_proporcional(cadera_contorno, cadera, cadera_width, 1)
    
    # Obtener el ancho máximo de todas las imágenes
    max_width = max(cabeza_contorno.shape[1], cuello_contorno.shape[1], pecho_contorno.shape[1],
                    cintura_contorno.shape[1], cadera_contorno.shape[1])

    # Rellenar las imágenes más estrechas para que tengan el mismo ancho que la más ancha
    cabeza_contorno = rellenar_ancho(cabeza_contorno, max_width)
    cuello_contorno = rellenar_ancho(cuello_contorno, max_width)
    pecho_contorno = rellenar_ancho(pecho_contorno, max_width)
    cintura_contorno = rellenar_ancho(cintura_contorno, max_width)
    cadera_contorno = rellenar_ancho(cadera_contorno, max_width)

    # Concatenar verticalmente los contornos
    imagen_completa = np.vstack((cabeza_contorno, cuello_contorno, pecho_contorno,
                                 cintura_contorno, cadera_contorno))

    cv2.imshow("A", imagen_completa)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    results = list()

    results.append({
        "url": cabeza_contorno
    })
    
    results.append({
        "url": cuello_contorno
    })

    results.append({
        "url": pecho_contorno
    })

    results.append({
        "url": cintura_contorno
    })

    results.append({
        "url": cadera_contorno
    })
    
    results.append({
        "url": piernas_contorno
    })


    return jsonify({"status": True, "results": results })

def remove_background(path_archivo):
    img = cv2.imread(path_archivo)
    hh, ww = img.shape[:2]
    # threshold on white
    # Define lower and uppper limits
    lower = np.array([200, 200, 200])
    upper = np.array([255, 255, 255])

    # Create mask to only select black
    thresh = cv2.inRange(img, lower, upper)

    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # invert morp image
    mask = 255 - morph

    # apply mask to image
    result = cv2.bitwise_and(img, img, mask=mask)


    # save results
    # cv2.imwrite('pills_thresh.jpg', thresh)
    # cv2.imwrite('pills_morph.jpg', morph)
    # cv2.imwrite('pills_mask.jpg', mask)
    cv2.imwrite('result.jpg', result)
    """
    cv2.imshow('thresh', thresh)
    cv2.imshow('morph', morph)
    cv2.imshow('mask', mask)
    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    return 'result.jpg'


def draw_inside(file):
    imagen = cv2.imread('pink.png')
    # Convertir la imagen de BGR a HSV
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    # Definir el rango de color para el blanco
    rango_blanco_bajo = np.array([0, 0, 200])
    rango_blanco_alto = np.array([180, 30, 255])

    # Crear una máscara para el color blanco
    mascara_blanco = cv2.inRange(hsv, rango_blanco_bajo, rango_blanco_alto)

    # Invertir la máscara para excluir el color blanco
    mascara_excluida = cv2.bitwise_not(mascara_blanco)

    # Aplicar operaciones morfológicas para mejorar la máscara excluida
    kernel = np.ones((5, 5), np.uint8)
    mascara_excluida = cv2.morphologyEx(mascara_excluida, cv2.MORPH_OPEN, kernel)

    # Encontrar los contornos en la máscara excluida
    contornos, _ = cv2.findContours(mascara_excluida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # Crear una imagen en blanco del mismo tamaño que la original
    imagen_contorno = np.zeros_like(imagen)

    # Dibujar los contornos en la imagen de contorno
    cv2.drawContours(imagen_contorno, contornos, -1, (0, 255, 0), 2)

    cv2.imwrite('imagen_contorno.jpg', imagen_contorno)

    #Mostrar solo el contorno
    cv2.imshow('Contorno', imagen_contorno)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #transform('imagen_contorno.jpg')


def transform(path_archivo):
    contorno = cv2.imread(path_archivo)
    y_min = np.min(contorno[:, 0, 1])
    y_max = np.max(contorno[:, 0, 1])
    y_medio = int((y_min + y_max) / 2)

    # Divide el contorno en dos partes en el punto medio vertical
    parte_superior = contorno[contorno[:, 0, 1] < y_medio]
    parte_inferior = contorno[contorno[:, 0, 1] >= y_medio]

    # Calcula la distancia entre los puntos más alejados en cada mitad del contorno
    distancia_superior = np.max(parte_superior[:, 0, 0]) - np.min(parte_superior[:, 0, 0])
    distancia_inferior = np.max(parte_inferior[:, 0, 0]) - np.min(parte_inferior[:, 0, 0])

    # Define el factor de aumento del ancho
    factor_aumento = 1.5

    # Aumenta la distancia entre los puntos más alejados en cada mitad del contorno
    nueva_distancia_superior = int(distancia_superior * factor_aumento)
    nueva_distancia_inferior = int(distancia_inferior * factor_aumento)

    # Ajusta los puntos extremos en cada mitad del contorno según el nuevo ancho deseado
    parte_superior[:, 0, 0] = parte_superior[:, 0, 0] - int((nueva_distancia_superior - distancia_superior) / 2)
    parte_inferior[:, 0, 0] = parte_inferior[:, 0, 0] - int((nueva_distancia_inferior - distancia_inferior) / 2)

    # Une las dos mitades del contorno nuevamente
    contorno_modificado = np.concatenate((parte_superior, parte_inferior[::-1]))

    # Dibuja el contorno modificado en la imagen original (sustituye 'imagen' con tu imagen real)
    imagen = cv2.imread('ruta_a_la_imagen.jpg') 
    cv2.drawContours(imagen, [contorno_modificado], 0, (0, 255, 0), 2)

    # Muestra la imagen con el contorno modificado
    cv2.imshow('Contorno Modificado', imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def crop(path_archivo):
    o_img = cv2.imread(path_archivo, cv2.IMREAD_UNCHANGED)
    
    print('Original Dimensions : ',o_img.shape)
    
    width = 557
    height = 837
    dim = (width, height)
    
    # resize image
    img = cv2.resize(o_img, dim, interpolation = cv2.INTER_AREA)


    # img=cv2.imread('reference3.png')
    # Prints Dimensions of the image
    # print(img.shape)

    print(img.shape) 
    
    #cabeza
    cabeza_a = img[0:150, 174:262] # Slicing to crop the image 
    cv2.imwrite('cabezaA.jpg', cabeza_a)

    cabeza_b = img[0:150, 263:350] # Slicing to crop the image 
    cv2.imwrite('cabezaB.jpg', cabeza_b)


    #pecho
    pecho_a = img[150:200, 174:262] # Slicing to crop the image 
    cv2.imwrite('pechoA.jpg', pecho_a)
     
    pecho_b = img[150:200, 263:350] # Slicing to crop the image 
    cv2.imwrite('pechoB.jpg', pecho_b)
    
    #torso
    torso_a = img[150:350, 174:262] # Slicing to crop the image 
    cv2.imwrite('torsoA.jpg', torso_a)


    torso_b = img[150:350, 263:350] # Slicing to crop the image 
    cv2.imwrite('torsoB.jpg', torso_b)
    
    #piernas
    pierna_a = img[450:900, 174:262] # Slicing to crop the image 
    cv2.imwrite('piernaA.jpg', pierna_a)

    pierna_b = img[450:900, 263:350] # Slicing to crop the image 
    cv2.imwrite('piernaB.jpg', pierna_b)


    results = list()
    results.append(upload('cabezaA.jpg'))
    results.append(upload('cabezaB.jpg'))
    results.append(upload('pechoA.jpg'))
    results.append(upload('pechoB.jpg'))
    results.append(upload('torsoA.jpg'))
    results.append(upload('torsoB.jpg'))
    results.append(upload('piernaA.jpg'))
    results.append(upload('piernaB.jpg'))

    return jsonify({"status": True, "results": results })




def generate_shape_by_weight(data):
    img_url = data['image_url']
    peso = data['peso']
    
    print(f"img_url {img_url}")

    # Realiza la descarga de la imagen desde Cloudinary
    response = requests.get(img_url)
    ruta_destino = 'temp.jpg'

    # Verifica si la descarga fue exitosa
    if response.status_code == 200:
        # Guarda la imagen en el servidor
        with open(ruta_destino, 'wb') as file:
            file.write(response.content)
    else:
        print('No se pudo descargar la imagen desde la URL de Cloudinary')
        return jsonify({"status": False, "message": "No se pudo descargar" })

    imagen = cv2.imread('temp.jpg')   
    results = list()
    
    
    cv2.imwrite('original.jpg', imagen)
    results.append({
        "name": "original.jpg",
        "url": upload("original.jpg")
    })

    # Convertir la imagen de BGR a HSV
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    # Definir el rango de color para el blanco
    rango_blanco_bajo = np.array([0, 0, 200])
    rango_blanco_alto = np.array([180, 30, 255])

    # Crear una máscara para el color blanco
    mascara_blanco = cv2.inRange(hsv, rango_blanco_bajo, rango_blanco_alto)

    # Invertir la máscara para excluir el color blanco
    mascara_excluida = cv2.bitwise_not(mascara_blanco)

    # Aplicar operaciones morfológicas para mejorar la máscara excluida
    kernel = np.ones((5, 5), np.uint8)
    mascara_excluida = cv2.morphologyEx(mascara_excluida, cv2.MORPH_OPEN, kernel)

    # Encontrar los contornos en la máscara excluida
    contornos, _ = cv2.findContours(mascara_excluida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # Crear una imagen en blanco del mismo tamaño que la original
    imagen_contorno = np.zeros_like(imagen)

    # Dibujar los contornos en la imagen de contorno
    cv2.drawContours(imagen_contorno, contornos, -1, (0, 255, 0), 2)

    cv2.imwrite('imagen_contorno.jpg', imagen_contorno)



    contour = contornos[0]
    # Dibuja el contorno original y el contorno reducido en una imagen
    image = np.ones((500, 500, 3), dtype=np.uint8)  * 255  # Crea una imagen en blanco para dibujar
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)  # Dibuja el contorno original en verde

    # Muestra la imagen con los contornos dibujados
    # cv2.imshow('Contornos', image)
    cv2.imwrite('contorno.jpg', image)    
    results.append({
        "name": 'contorno.jpg',
        "url": upload(f'contorno.jpg')
    })

    clean_value = int(peso)/-80 if int(peso)<80 else int(peso)/80
    reduction_factor = 1+(0.1*clean_value)  # Ajusta este valor según tus necesidades

    # Calcula el punto medio del contorno en el eje x
    x_mean = np.mean(contour[:, 0, 0])

    # Calcula el desplazamiento basado en el punto medio y el factor de reducción
    shift = (1 - reduction_factor) * x_mean

    # Aplica la reducción horizontal al contorno
    reduced_contour = contour.copy()  # Copia el contorno original
    reduced_contour[:, 0, 0] = (reduced_contour[:, 0, 0] - shift) * reduction_factor + shift

        
    image_2 = np.ones((500, 800, 3), dtype=np.uint8) * 255  # Crea una imagen en blanco para dibujar
    cv2.drawContours(image_2, [reduced_contour], -1, (0, 0, 255), 2)  
        
    filename = f'new_weight.jpg'
    cv2.imwrite(filename, image_2)
        
    results.append({
        "name": filename,
        "url": upload(filename)
    })


    return jsonify({"status": True, "url_images": results })


def upload(path_archivo):
    list_images = list()
    try:
        with open(path_archivo, 'rb') as image_file:
            #image_file = files.get(image)
            upload = upload_image(image_file)
            #transform = transform_image(upload, '350', '350')
            #url_image = transform[0].replace("http", "https")
            list_images.append(upload['secure_url'])
        return list_images[0]
        #return jsonify({"status": True, "url_images": list_images})
    except Exception as e:
        print(f"ERROR!!!! {e}")
        return False
        #return jsonify({"status": False, "message": "some wrong"})
    


# Función para escalar una imagen horizontalmente
def escalar_horizontal(imagen, nuevo_ancho):
    altura, ancho = imagen.shape[:2]
    escala_horizontal = nuevo_ancho / ancho
    return cv2.resize(imagen, (nuevo_ancho, int(altura * escala_horizontal)))



def ad(anchos_partes=[200, 250, 100, 400], ancho_total=2000, num_partes=4):
    # Ruta de la imagen a procesar
    ruta_imagen = 'original.jpg'

    # Cargar la imagen
    imagen_original = cv2.imread(ruta_imagen)

    # Pedir al usuario el ancho total de la imagen
    ancho_total = int(input("Ingresa el ancho total de la imagen: "))

    # Pedir al usuario los anchos individuales de cada parte
    num_partes = int(input("Ingresa el número de partes: "))
    anchos_partes = []
    for i in range(num_partes):
        ancho = int(input(f"Ingresa el ancho de la parte {i + 1}: "))
        anchos_partes.append(ancho)

    # Lista para almacenar las partes escaladas horizontalmente
    partes_escaladas = []

    # Escalar cada parte horizontalmente
    for ancho_parte in anchos_partes:
        parte_escalada = escalar_horizontal(imagen_original, ancho_parte)
        partes_escaladas.append(parte_escalada)

    # Ajustar el tamaño de la imagen final al ancho total ingresado por el usuario
    altura = partes_escaladas[0].shape[0]
    imagen_completa = np.zeros((altura, ancho_total, 3), dtype=np.uint8)

    # Combinar las partes escaladas en la imagen completa
    inicio = 0
    for parte_escalada in partes_escaladas:
        fin = inicio + parte_escalada.shape[1]
        imagen_completa[:, inicio:fin] = parte_escalada
        inicio = fin

    # Mostrar la imagen resultante con los espacios llenos
    cv2.imshow('Imagen Completa', imagen_completa)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def escalar_proporcional(imagen, nuevo_valor, referencia, eje):
    # Escala la imagen proporcionalmente según el nuevo valor y la referencia
    proporcion = nuevo_valor / referencia
    if eje == "x":
        return cv2.resize(imagen, (int(imagen.shape[1] * proporcion), imagen.shape[0]))
    elif eje == "y":
        return cv2.resize(imagen, (imagen.shape[1], int(imagen.shape[0] * proporcion)))
    else:
        return imagen

def ajustar_medidas_cuerpo(image_parts, medidas):
    # Cargar las imágenes de las diferentes partes del cuerpo
    images = {part: cv2.imread(image_path) for part, image_path in image_parts.items()}

    # Ajustar las medidas proporcionales de cada parte del cuerpo según las medidas ingresadas por el usuario
    medida_original_cintura = images["cintura"].shape[0]  # Valor de ejemplo, deberías obtener esta medida de la imagen original
    medida_original_pecho = images["pecho"].shape[0]  # Valor de ejemplo, deberías obtener esta medida de la imagen original
    medida_original_cadera = images["cadera"].shape[0]  # Valor de ejemplo, deberías obtener esta medida de la imagen original


    medida_original_cuello = 80

    medida_original_piernas = 80

    
    images["cintura"] = escalar_proporcional(images["cintura"], medidas["cintura"], medida_original_cintura, "x")
    images["pecho"] = escalar_proporcional(images["pecho"], medidas["pecho"], medida_original_pecho, "x")
    images["cadera"] = escalar_proporcional(images["cadera"], medidas["cadera"], medida_original_cadera, "x")

    images["cuello"] = escalar_proporcional(images["cuello"], medidas["cuello"], medida_original_cuello, "x")
    images["piernas"] = escalar_proporcional(images["piernas"], medidas["piernas"], medida_original_piernas, "x")


    # Obtener el ancho máximo de todas las imágenes
    max_width = max(img.shape[1] for img in images.values())

    # Redimensionar las imágenes más pequeñas para que tengan el mismo ancho que la imagen más grande
    for part, img in images.items():
        if img.shape[1] < max_width:
            resize_ratio = max_width / img.shape[1]
            images[part] = cv2.resize(img, (max_width, int(img.shape[0] * resize_ratio)))

    # Combinar las imágenes de las diferentes partes para formar la imagen completa del cuerpo ajustada
    combined_image = cv2.vconcat([images["cabeza"], images["cuello"], images["pecho"],
                                images["cintura"], images["cadera"], images["piernas"]])

    # Mostrar la imagen completa del cuerpo ajustada
    """
    cv2.imshow("Cuerpo ajustado", combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    gris = cv2.cvtColor(combined_image, cv2.COLOR_BGR2GRAY)
    # Aplicar suavizado Gaussiano
    gauss = cv2.GaussianBlur(gris, (5,5), 0)
    #cv2.imshow("suavizado", gauss)
    # Detectamos los bordes con Canny
    canny = cv2.Canny(gauss, 50, 150)
    # Buscamos los contornos
    (contornos,_) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fondo = np.ones_like(combined_image) * 255
    cv2.drawContours(fondo, contornos, -1, (0, 0, 0), 2)
    
    cv2.imwrite("result.jpg", fondo)
    """
    cv2.imshow("contornos", fondo)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    return jsonify({"status": True, "result": upload("result.jpg")})

def dividir_imagen(image_path):
    # Cargar la imagen del cuerpo humano
    image = cv2.imread(image_path)

    # Dividir la imagen en diferentes partes
    head = image[0:120, :]
    neck = image[121:180, :]
    chest = image[181:350, :]
    waist = image[351:430, :]
    hip = image[431:530, :]
    legs = image[531:, :]

    # Mostrar las imágenes divididas
    """
    cv2.imshow("Cabeza", head)
    cv2.imshow("Cuello", neck)
    cv2.imshow("Pecho", chest)
    cv2.imshow("Cintura", waist)
    cv2.imshow("Cadera", hip)
    cv2.imshow("Piernas", legs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    
    cv2.imwrite("cabeza.jpg", head)
    cv2.imwrite("cuello.jpg", neck)
    cv2.imwrite("pecho.jpg", chest)
    cv2.imwrite("cintura.jpg", waist)
    cv2.imwrite("cadera.jpg", hip)
    cv2.imwrite("piernas.jpg", legs)
    