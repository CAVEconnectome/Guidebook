from flask import Flask
from flask_cors import CORS
from .config import configure_app
from . import blueprint
import os

__version__ = blueprint.__version__
here = os.path.dirname(__file__)


def create_app():
    app = Flask(__name__,
                static_url_path='/guidebook/static',
                static_folder=f'static')
    app = configure_app(app)
    CORS(app, expose_headers=['WWW-Authenticate', 'X-Requested-With'])
    app.register_blueprint(blueprint.bp)
    return app
