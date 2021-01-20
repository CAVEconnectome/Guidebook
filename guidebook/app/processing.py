from flask import Blueprint, redirect, request, flash, render_template, url_for, current_app, jsonify
from ..base import generate_proofreading_state, generate_lvl2_proofreading
from .forms import SkeletonizeForm, Lvl2SkeletonizeForm
import numpy as np
from middle_auth_client import auth_requires_permission, auth_required
import re
from annotationframeworkclient import FrameworkClient
from rq import Queue, Retry
from rq.job import Job
from .worker import conn

api_version = 0
url_prefix = f"/guidebook"
api_prefix = f"/api/v{api_version}"
bp = Blueprint("guidebook", __name__, url_prefix=url_prefix)

__version__ = "0.0.1"
DEFAULT_DATASTACK = 'minnie65_phase3_v1'

q = Queue(connection=conn)


@bp.route("/")
def index():
    return f"Neuron Guidebook v. {__version__}"


@auth_required
@bp.route("/landing")
def landing_page():
    return render_template('landing.html', title='Neuron Guidebook')


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


# @auth_requires_permission("view")
# @bp.route(f"{api_prefix}/datastack/<datastack>/root_id/<int:root_id>/skeletonize", methods=["GET", "POST"])
# def generate_guidebook(datastack, root_id,):
#     root_is_soma = request.args.get('root_is_soma', False)
#     root_loc = parse_root_location(request.args.get('root_location', None))
#     if root_loc is not None:
#         root_loc = root_loc * [4, 4, 40]
#     state = generate_proofreading_state(
#         datastack, int(root_id), root_is_soma=root_is_soma, root_loc=root_loc,  return_as='url')
#     return redirect(state, code=302)


@auth_requires_permission("view")
@bp.route(f"{api_prefix}/datastack/<datastack>/root_id/<int:root_id>/coarse_branch")
def generate_guidebook_chunkgraph(datastack, root_id):
    root_loc = parse_root_location(request.args.get('root_location', None))
    branch_points = request.args.get('branch_points', 'True') == 'True'
    end_points = request.args.get('end_points', 'True') == 'True'
    kwargs = {
        'return_as': 'url',
        'root_point': root_loc,
        'refine_branch_points': branch_points,
        'refine_end_points': end_points,
    }
    job = q.enqueue_call(generate_lvl2_proofreading, args=(datastack, int(root_id)),
                         kwargs=kwargs,
                         result_ttl=5000, timeout=600, retry=Retry(max=2, interval=30))
    return redirect(url_for('.show_skeletonization_result', job_key=job.get_id()))


@bp.route('/skeletonization/results/<job_key>')
def show_skeletonization_result(job_key):
    job = Job.fetch(job_key, connection=conn)
    if job.is_finished:
        return render_template('show_link.html', ngl_url=job.result)
    elif job.get_status() == "failed":
        return error_page(job.exc_info)
    else:
        return "Not done yet, refresh in a few seconds."


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
        point_option = form.point_option.data
        if point_option == 'both':
            branch_points = True
            end_points = True
        elif point_option == 'ep':
            branch_points = False
            end_points = True
        elif point_option == 'bp':
            branch_points = True
            end_points = False
        try:
            root_loc_formatted = encode_root_location(form.root_location.data)
        except Exception as e:
            return error_page(e)
        url = url_for('.generate_guidebook_chunkgraph',
                      datastack=datastack, root_id=root_id, root_location=root_loc_formatted, branch_points=branch_points, end_points=end_points)
        return redirect(url)

    return render_template('lvl2_skeletonize.html',
                           title='Neuron Guidebook',
                           form=form)
