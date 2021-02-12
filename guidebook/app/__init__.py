from flask import Flask
from .config import configure_app
from . import processing

__version__ = processing.__version__


def create_app():
    app = Flask(__name__,
                static_folder='../static')
    app = configure_app(app)
    app.register_blueprint(processing.bp)
    print(f'App is running datastack: {app.config.get("DATASTACK")}')
    return app
