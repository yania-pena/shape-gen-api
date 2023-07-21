from api.application.routes import image_routes, status_routes
from flask import Blueprint

application = Blueprint(
    'application',
    __name__
)