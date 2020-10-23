from flask import Blueprint, redirect, request
from ..base import generate_proofreading_state
import numpy as np

bp = Blueprint("guidebook", __name__, url_prefix="/guidebook/v0")

__version__ = "0.0.1"


@bp.route("/")
def index():
    return f"Guidebook v. {__version__}"


@bp.route("/datastack/<datastack>/root_id/<root_id>", methods=["GET", "POST"])
def generate_guidebook(datastack, root_id):
    root_is_soma = request.args.get('root_is_soma', False)
    root_loc = request.args.get('root_loc', None)
    if root_loc is not None:
        root_loc = np.array(root_loc.split('_')).astype(int) * [4, 4, 40]
    state = generate_proofreading_state(
        datastack, int(root_id), root_is_soma=root_is_soma, root_loc=root_loc,  return_as='url')
    return redirect(state, code=302)
