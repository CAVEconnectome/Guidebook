from flask import Blueprint, redirect, request, flash, render_template, url_for, current_app
from ..base import generate_proofreading_state, generate_lvl2_proofreading
from .forms import SkeletonizeForm, Lvl2SkeletonizeForm
import numpy as np
from middle_auth_client import auth_requires_permission, auth_required
import re
from annotationframeworkclient import FrameworkClient

api_version = 0
url_prefix = f"/guidebook"
api_prefix = f"/api/v{api_version}"
bp = Blueprint("guidebook", __name__, url_prefix=url_prefix)

__version__ = "0.0.1"
DEFAULT_DATASTACK = 'minnie65_phase3_v1'


@bp.route("/")
def index():
    return f"Neuron Guidebook v. {__version__}"


@auth_required
@bp.route("/landing")
def landing_page():
    return render_template('landing.html', title='Neuron Guidebook')


@auth_requires_permission("view")
@bp.route("skeletonize", methods=['GET', 'POST'])
def skeletonize_form():
    form = SkeletonizeForm()
    if form.validate_on_submit():
        datastack = current_app.config.get('DATASTACK', DEFAULT_DATASTACK)
        root_id = form.root_id.data
        root_loc_formatted = encode_root_location(form.root_location.data)
        values = {'root_is_soma': form.root_is_soma.data,
                  'root_loc': root_loc_formatted}
        url = url_for('.generate_guidebook',
                      datastack=datastack, root_id=root_id, **values)

        return redirect(url)

    return render_template('skeletonize.html',
                           title='Neuron Guidebook',
                           form=form)


def encode_root_location(form_data):
    if len(form_data) == 0:
        return None
    elif re.match(r"^( |\d)*,( |\d)*,( |\d)*$", form_data):
        root_loc_data = np.fromstring(
            form_data, count=3, dtype=np.int, sep=',')
        root_loc_formatted = '_'.join(map(str, root_loc_data))
        return root_loc_formatted
    else:
        raise ValueError(
            "Root location must be specified with 3 comma-separated numbers")


def parse_root_location(root_loc, field_name='root_loc'):
    if root_loc is not None:
        root_loc = np.array(root_loc.split('_')).astype(
            int)
        print(f'Root location is: {root_loc}')
    return root_loc


@auth_requires_permission("view")
@bp.route(f"{api_prefix}/datastack/<datastack>/root_id/<int:root_id>/skeletonize", methods=["GET", "POST"])
def generate_guidebook(datastack, root_id,):
    root_is_soma = request.args.get('root_is_soma', False)
    root_loc = parse_root_location(request.args.get('root_location', None))
    if root_loc is not None:
        root_loc = root_loc * [4, 4, 40]
    state = generate_proofreading_state(
        datastack, int(root_id), root_is_soma=root_is_soma, root_loc=root_loc,  return_as='url')
    return redirect(state, code=302)


@auth_requires_permission("view")
@bp.route(f"{api_prefix}/datastack/<datastack>/root_id/<int:root_id>/coarse_branch")
def generate_guidebook_chunkgraph(datastack, root_id):
    root_loc = parse_root_location(request.args.get('root_location', None))
    try:
        state = generate_lvl2_proofreading(
            datastack, int(root_id), return_as='url', root_point=root_loc)
        # short_url = upload_state(state)
        return redirect(state, 302)
    except Exception as e:
        return error_page(e)


def upload_state(state):
    client = FrameworkClient(DEFAULT_DATASTACK)
    sid = client.state.upload_state_json(state)
    return client.state.build_neuroglancer_url(sid)


def error_page(error):
    return render_template("error.html", error_text=error)


@auth_requires_permission("view")
@bp.route("coarse_skeletonize", methods=['GET', 'POST'])
def lvl2_form():
    form = Lvl2SkeletonizeForm()
    if form.validate_on_submit():
        datastack = DEFAULT_DATASTACK
        root_id = form.root_id.data
        try:
            root_loc_formatted = encode_root_location(form.root_location.data)
        except Exception as e:
            return error_page(e)
        url = url_for('.generate_guidebook_chunkgraph',
                      datastack=datastack, root_id=root_id, root_location=root_loc_formatted)
        return redirect(url)

    return render_template('lvl2_skeletonize.html',
                           title='Neuron Guidebook',
                           form=form)
