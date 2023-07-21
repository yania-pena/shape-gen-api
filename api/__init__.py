from flask import Flask
from flask_cors import CORS
import os.path
from flask_pymongo import PyMongo
import json
import datetime
from bson.objectid import ObjectId
from config import MONGO_URI
from flask_jwt_extended import JWTManager
from datetime import timedelta

class JSONEncoder(json.JSONEncoder):
    ''' extend json-encoder class'''

    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        if isinstance(o, datetime.datetime):
            return str(o)
        return json.JSONEncoder.default(self, o)


"""Create Flask application."""
app = Flask(__name__)
app.json_encoder = JSONEncoder
#app.config['MONGO_URI'] = MONGO_URI
#app.config["JWT_SECRET_KEY"] = "educational-app-789546546"
#app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=72)
#mongo = PyMongo(app)
#jwt_manager = JWTManager(app)
cors = CORS(app)


with app.app_context():
    # Register Blueprints
    from .application import application
    app.register_blueprint(application, url_prefix='/api/application')