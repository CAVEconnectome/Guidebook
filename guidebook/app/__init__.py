from flask import Flask
from .config import configure_app
from . import processing
import os

__version__ = processing.__version__
base_dir = os.path.join(os.path.dirname(__file__), '..')


def create_app():
    app = Flask(__name__,
                static_folder=f'{base_dir}/static')
    app = configure_app(app)
    app.register_blueprint(processing.bp)
    return app
