from flask import request
from api import app
from api.application.controllers import image_controller
import requests
import cv2

@app.route("/images/shape", methods=["POST"])
def generate_shape():
    file = request.files['imagen']
    return image_controller.new_method(file)


@app.route("/images/shape/transform_by_weight", methods=["POST"])
def transform_shape_by_weight():
    return image_controller.generate_shape_by_weight(request.json)


@app.route("/images/shape/transform", methods=["POST"])
def transform_shape():

    img_url = request.json["image_url"]

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


    image_controller.dividir_imagen("temp.jpg")
    
    image_parts = {
        "cabeza": "cabeza.jpg",
        "cuello": "cuello.jpg",
        "pecho": "pecho.jpg",
        "cintura": "cintura.jpg",
        "cadera": "cadera.jpg",
        "piernas": "piernas.jpg"
    }



    medidas = {
        "cintura": request.json["medida_cintura"],
        "pecho": request.json["medida_pecho"],
        "cadera": request.json["medida_cadera"],
        "cuello": request.json["medida_cuello"],
        "piernas": request.json["medida_piernas"]
    }


    return image_controller.ajustar_medidas_cuerpo(image_parts, medidas)


