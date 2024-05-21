#!/usr/bin/python3
""" Flask Application """
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS
from flask import make_response,jsonify

app = Flask(__name__)
app.url_map.strict_slashes = False

# Register blueprints
cors = CORS(app)

@app.route('/api/v1/status', methods= ['GET'])
def status():
    """ Status of users"""
    response_data = {"status": "OK"}
    response = make_response(jsonify(response_data), 200)
    return response

@app.errorhandler(404)
def not_found(error):
    """ 404 Error
    ---
    responses:
      404:
        description: a resource was not found
    """
        return make_response(jsonify({'error': "Not found"}), 404)
