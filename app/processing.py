from flask import Blueprint, redirect
from guidebook.base import generate_proofreading_state

bp = Blueprint("guidebook", __name__, url_prefix="/guidebook/v0")

__version__ = "0.0.1"


@bp.route("/")
def index():
    return f"Guidebook v. {__version__}"


@bp.route("/datastack/<datastack>/root_id/<root_id>", methods=["GET"])
def generate_guidebook(datastack, root_id):
    state = generate_proofreading_state(
        datastack, int(root_id), return_as='url')
    return redirect(state, code=302)
