import os
import flask
from flask_caching import Cache
from flask import Blueprint, render_template, request
from middle_auth_client import auth_required
from ..tourguide_lib.lib_utils import make_global_client, ServerIncompatibilityError
from .config import TOURGUIDE_PREFIX

api_bp = Blueprint("main", __name__)
__version__ = "2.3.3"

cache = Cache(
    config={
        "CACHE_TYPE": "SimpleCache",
        "CACHE_DEFAULT_TIMEOUT": 24 * 3600,
    }
)


def has_service_key(datastack_name, gclient):
    return f"has_skeleton_service_{datastack_name}"


@cache.cached(make_cache_key=has_service_key)
def has_skeleton_service(datastack_name, gclient):
    info = gclient.info.get_datastack_info(datastack_name=datastack_name)
    return info.get("skeleton_source") is not None


def mirror_key(datastack_name, gclient):
    return f"get_image_mirrors_{datastack_name}"


@cache.cached(make_cache_key=mirror_key)
def get_image_mirrors(datastack_name, gclient):
    try:
        mirrors = gclient.info.get_image_mirror_names(datastack_name=datastack_name)
    except ServerIncompatibilityError:
        mirrors = []
    return mirrors


@api_bp.route("/")
@auth_required
def index():
    gclient = make_global_client(
        server_address=os.environ.get("TOURGUIDE_SERVER_ADDRESS"),
        auth_token=flask.g.get("auth_token"),
    )
    datastacks = gclient.info.get_datastacks()
    show_datastacks = sorted(
        [ds for ds in datastacks if has_skeleton_service(ds, gclient)]
    )
    mirrors = {ds: get_image_mirrors(ds, gclient) for ds in datastacks}
    base_url = f"{request.url_root.rstrip('/')}/{TOURGUIDE_PREFIX.strip('/')}"
    return render_template(
        "index.html",
        title="TourGuide",
        version=__version__,
        base_url=base_url,
        datastacks=show_datastacks,
        mirrors=mirrors,
    )


@api_bp.route("/version")
def version():
    return __version__
