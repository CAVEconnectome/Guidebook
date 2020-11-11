import os
import json


class BaseConfig(object):
    SECRET_KEY = os.environ.get('GUIDEBOOK_CSRF_KEY', 'test_key_1293473')
    DATASTACK = 'minnie65_phase3_v1'
    DEBUG = True

    if os.environ.get("DAF_CREDENTIALS", None) is not None:
        with open(os.environ.get("DAF_CREDENTIALS"), "r") as f:
            AUTH_TOKEN = json.load(f)["token"]
    else:
        AUTH_TOKEN = None


def configure_app(app):
    app.config.from_object(BaseConfig)
    return app
