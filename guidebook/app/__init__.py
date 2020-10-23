from flask import Flask
from . import processing


def create_app():
    app = Flask(__name__)
    app.register_blueprint(processing.bp)
    return app
