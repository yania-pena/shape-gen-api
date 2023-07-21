# shape-gen-api
Requisitos:
- Python 3.8.9
- Virtualenv

# Instalar virtualenv 
- Dentro de una terminal ejecutar `pip install virtualenv`

# Instalación del proyecto localmente
- Clonar el repositorio
- Dentro de la carpeta del repositorio ejecutar `python -m venv venv/`
- El comando anterior debe crear un carpeta llamada venv.
- Ejecutar `source venv/bin/activate`(Linux) o  `.\venv\Scripts\activate`(Windows) .
- Ejecutar `pip install -r requirements.txt`

# Ejecutar proyecto
- Activar entorno virtual, para esto es necesario ejecutar este comando `source venv/bin/activate`(linux) o `.\venv\Scripts\activate`(si ya se ejecutó antes no es necesario volverlo a hacer)
- Ejecutar `python run.py`


# Probar los endpoints localmente
Originalmente el api se levanta en esta dirección: http://127.0.0.1:5000 o su equivalente http://localhost:5000, por lo tanto los endpoints son así:
Ejemplo
http://localhost:5000/shape-gen-api
