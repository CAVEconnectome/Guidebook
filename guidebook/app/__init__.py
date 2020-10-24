from flask import Flask
from .config import BaseConfig
from . import processing


def create_app():
    app = Flask(__name__)
    app.config.from_object(BaseConfig)
    app.register_blueprint(processing.bp)
    return app
