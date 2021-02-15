from flask import Flask
from .config import configure_app
from . import processing
import os

__version__ = processing.__version__
here = os.path.dirname(__file__)


def create_app():
    app = Flask(__name__,
                static_folder=f'{here}/static')
    app = configure_app(app)
    app.register_blueprint(processing.bp)
    return app
