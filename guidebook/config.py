import os
import json


class BaseConfig(object):
    SECRET_KEY = os.environ.get("GUIDEBOOK_CSRF_KEY", "test_key_1293473")
    DATASTACK = os.environ.get("GUIDEBOOK_DATASTACK")
    N_PARALLEL = os.environ.get("GUIDEBOOK_N_PARALLEL", 1)
    INVALIDATION_D = os.environ.get("GUIDEBOOK_INVALIDATION_D", 3)
    GLOBAL_SERVER_ADDRESS = os.environ.get("GLOBAL_SERVER_ADDRESS", None)
    GUIDEBOOK_EXPECTED_RESOLUTION = os.environ.get(
        "GUIDEBOOK_EXdPECTED_RESOLUTION", "4,4,40"
    )
    GUIDEBOOK_EXPECTED_RESOLUTION = [
        r for r in map(float, GUIDEBOOK_EXPECTED_RESOLUTION.split(","))
    ]
    AUTH_TOKEN_KEY = os.environ.get("AUTH_TOKEN_KEY", "token")
    if os.environ.get("DAF_CREDENTIALS", None) is not None:
        with open(os.path.expanduser(os.environ.get("DAF_CREDENTIALS")), "r") as f:
            AUTH_TOKEN = json.load(f)[AUTH_TOKEN_KEY]
    else:
        AUTH_TOKEN = None
    SHOW_PATH_TOOL = os.environ.get("GUIDEBOOK_SHOW_PATH_TOOL", "false") == "true"
    SHORT_SEGMENT_THRESH = os.environ.get("GUIDEBOOK_SHORT_SEGMENT_THRESH", 15_000)
    USE_L2CACHE = os.environ.get("USE_L2CACHE", "false") == "true"
    EP_PROOFREADING_TAGS = ["checked", "error", "correct"]
    BP_PROOFREADING_TAGS = ["checked", "error"]
    CONTRAST_LOOKUP = {
        "minnie65_phase3_v1": {"black": 0.35, "white": 0.7},
        "v1dd": {"black": 0.35, "white": 0.7},
    }


def configure_app(app):
    app.config.from_object(BaseConfig)
    return app
