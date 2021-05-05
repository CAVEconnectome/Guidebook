import os
import json


class BaseConfig(object):
    SECRET_KEY = os.environ.get("GUIDEBOOK_CSRF_KEY", "test_key_1293473")
    DATASTACK = os.environ.get("GUIDEBOOK_DATASTACK")
    N_PARALLEL = os.environ.get("GUIDEBOOK_N_PARALLEL", 1)
    INVALIDATION_D = os.environ.get("GUIDEBOOK_INVALIDATION_D", 3)
    GLOBAL_SERVER_ADDRESS = os.environ.get("GLOBAL_SERVER_ADDRESS", None)
    NEUROGLANCER_RESOLUTION = os.environ.get("NEUROGLANCER_RESOLUTION", None)

    if os.environ.get("DAF_CREDENTIALS", None) is not None:
        with open(os.environ.get("DAF_CREDENTIALS"), "r") as f:
            AUTH_TOKEN = json.load(f)["token"]
    else:
        AUTH_TOKEN = None


def configure_app(app):
    app.config.from_object(BaseConfig)
    return app
