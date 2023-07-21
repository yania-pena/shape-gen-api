from flask import jsonify
from api import app


@app.route("/status")
def status():
    return jsonify({"status": True, "greeting": "Hello friend"})
    