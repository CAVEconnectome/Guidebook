from flask import Blueprint, redirect, request, render_template, url_for, current_app
from .base import generate_lvl2_proofreading, generate_lvl2_paths
from .forms import Lvl2PathForm, Lvl2PointForm
from .utils import make_client, make_global_client
import numpy as np
from middle_auth_client import auth_required
import re
from rq import Queue, Retry
from rq.job import Job
from .worker import conn

api_version = 1
url_prefix = "/guidebook"
api_prefix = f"/api/v{api_version}"


bp = Blueprint("guidebook", __name__, url_prefix=url_prefix)
__version__ = "0.3.2"

q = Queue(connection=conn)


class GuidebookException(Exception):
    pass


@bp.route("/version")
def version_text():
    return f"Neuron Guidebook v.{__version__}"


@bp.route("/")
@auth_required()
def landing_page():
    client = make_global_client(
        server_address=current_app.config.get("GLOBAL_SERVER_ADDRESS"),
    )
    datastacks = client.info.get_datastacks()
    return render_template(
        "index.html",
        title="Neuron Guidebook",
        datastacks= sorted(datastacks),
        version=__version__,
    )


@bp.route(f"datastack/<datastack>/")
@auth_required
def datastack_page(datastack):
    return render_template(
        "datastack_page.html",
        title="Neuron Guidebook",
        datastack=datastack,
        show_path_tool=current_app.config.get("SHOW_PATH_TOOL", False),
        version=__version__,
    )


def encode_root_location(form_data):
    if len(form_data) == 0:
        return None
    regex_pattern = "(\d+)[^0-9]*(\d+)[^0-9]*(\d+)"  # Any three numbers separated by any non-digit characters.
    grp = re.search(regex_pattern, form_data)
    if grp is not None:
        return "_".join(grp.groups())
    else:
        raise ValueError("Please specify a 3d root location")


def parse_location(root_loc):
    if root_loc is not None:
        root_loc = np.array(root_loc.split("_")).astype(int)
    return root_loc


@bp.route(f"{api_prefix}/datastack/<datastack>/l2skeleton")
@auth_required
def generate_guidebook_chunkgraph(datastack):
    root_id = request.args.get("root_id", None)
    if root_id is not None:
        root_id = int(root_id)
    root_loc = parse_location(request.args.get("root_location", None))
    branch_points = request.args.get("branch_points", "True") == "True"
    end_points = request.args.get("end_points", "True") == "True"
    collapse_soma = request.args.get("collapse_soma") == "True"
    segmentation_fallback = request.args.get("segmentation_fallback", False) == "True"
    split_loc = parse_location(request.args.get("split_location", None))
    downstream = request.args.get("downstream") == "True"
    root_id_from_point = request.args.get("root_id_from_point") == "True"

    root_point_resolution = current_app.config.get(
        "GUIDEBOOK_EXPECTED_RESOLUTION", [1, 1, 1]
    )
    print(f"Resolution: {root_point_resolution}")

    kwargs = {
        "datastack": datastack,
        "server_address": current_app.config.get("GLOBAL_SERVER_ADDRESS"),
        "return_as": "short",
        "root_id": root_id,
        "root_point": root_loc,
        "refine_branch_points": branch_points,
        "refine_end_points": end_points,
        "collapse_soma": collapse_soma,
        "n_parallel": int(current_app.config.get("N_PARALLEL")),
        "root_point_resolution": root_point_resolution,
        "segmentation_fallback": segmentation_fallback,
        "invalidation_d": int(current_app.config.get("INVALIDATION_D")),
        "selection_point": split_loc,
        "downstream": downstream,
        "root_id_from_point": root_id_from_point,
        "auth_token_key": current_app.config.get("AUTH_TOKEN_KEY"),
        "l2cache": current_app.config.get("USE_L2CACHE", False),
        "ep_tags": current_app.config.get("EP_PROOFREADING_TAGS", []),
        "bp_tags": current_app.config.get("BP_PROOFREADING_TAGS", []),
        "contrast_lookup": current_app.config.get("CONTRAST_LOOKUP", {}),
        "cv_use_https": current_app.config.get("CV_USE_HTTPS", True),
    }
    print(kwargs)
    job = q.enqueue_call(
        generate_lvl2_proofreading,
        kwargs=kwargs,
        result_ttl=5000,
        timeout=600,
        retry=Retry(max=2, interval=10),
    )
    return redirect(url_for(".show_result_points", job_key=job.get_id()))


@bp.route(f"results/points/<job_key>")
@auth_required
def show_result_points(job_key):
    reload_time = 10
    try:
        job = Job.fetch(job_key, connection=conn)
        if job.is_finished:
            return render_template(
                "show_link.html",
                state_name="Branch and End Point",
                ngl_url=job.result,
                version=__version__,
            )
        elif job.get_status() == "failed":
            return error_page(job.exc_info)
        else:
            return wait_page(reload_time)
    except Exception as e:
        return error_page(str(e))


@bp.route(f"results/paths/<job_key>")
@auth_required
def show_result_paths(job_key):
    reload_time = 20
    try:
        job = Job.fetch(job_key, connection=conn)
        if job.is_finished:
            return render_template(
                "show_link.html",
                state_name="Path Review",
                ngl_url=job.result,
                version=__version__,
            )
        elif job.get_status() == "failed":
            return error_page(job.exc_info)
        else:
            return wait_page(reload_time)
    except Exception as e:
        return error_page(str(e))


def error_page(error):
    return render_template("error.html", error_text=error, version=__version__)


def wait_page(reload_time):
    return render_template(
        "job_wait.html", reload_time=reload_time, version=__version__
    )


@bp.route(f"datastack/<datastack>/skeletonize", methods=["GET", "POST"])
@auth_required
def lvl2_point_form(datastack):
    form = Lvl2PointForm()
    if form.validate_on_submit():
        root_id = form.root_id.data
        if len(root_id) == 0:
            root_id = None
        point_option = form.point_option.data
        segmentation_fallback = form.segmentation_fallback.data
        if point_option == "both":
            branch_points = True
            end_points = True
        elif point_option == "ep":
            branch_points = False
            end_points = True
        elif point_option == "bp":
            branch_points = True
            end_points = False
        try:
            root_loc_formatted = encode_root_location(form.root_location.data)
        except Exception as e:
            return error_page(e)
        root_is_soma = form.root_is_soma.data

        split_location_formatted = encode_root_location(form.split_location.data)
        if form.split_option.data == "downstream":
            downstream = True
        else:
            downstream = False
        root_id_from_point = form.root_id_from_root_loc.data

        url = url_for(
            ".generate_guidebook_chunkgraph",
            datastack=datastack,
            root_id=root_id,
            root_location=root_loc_formatted,
            branch_points=branch_points,
            end_points=end_points,
            collapse_soma=root_is_soma,
            segmentation_fallback=segmentation_fallback,
            split_location=split_location_formatted,
            downstream=downstream,
            root_id_from_point=root_id_from_point,
        )
        return redirect(url)

    return render_template(
        "lvl2_points.html",
        title="Neuron Guidebook",
        form=form,
        version=__version__,
        allow_segmentation=current_app.config.get("ALLOW_SEGMENTATION", False),
    )


@bp.route(f"{api_prefix}/datastack/<datastack>/l2paths")
@auth_required
def generate_guidebook_paths(datastack):
    if not current_app.config.get("SHOW_PATH_TOOL", False):
        raise GuidebookException("Path tool is not enabled")

    root_id = request.args.get("root_id", None)
    if root_id is not None:
        root_id = int(root_id)
    root_loc = parse_location(request.args.get("root_location", None))
    collapse_soma = request.args.get("collapse_soma") == "True"

    split_loc = parse_location(request.args.get("split_location", None))
    downstream = request.args.get("downstream") == "True"
    root_id_from_point = request.args.get("root_id_from_point") == "True"
    spacing = request.args.get("spacing", 3000)

    target_length = request.args.get("target_length", None)
    if target_length is not None:
        target_length = int(target_length)

    exclude_short = request.args.get("exclude_short", "True") == "True"
    if exclude_short:
        segment_length_thresh = current_app.config.get("SHORT_SEGMENT_THRESH", 2_000)
    else:
        segment_length_thresh = 0

    root_point_resolution = current_app.config.get(
        "GUIDEBOOK_EXPECTED_RESOLUTION", [1, 1, 1]
    )
    print(f"Resolution: {root_point_resolution}")

    kwargs = {
        "datastack": datastack,
        "server_address": current_app.config.get("GLOBAL_SERVER_ADDRESS"),
        "root_id": root_id,
        "root_point": root_loc,
        "root_point_resolution": root_point_resolution,
        "n_choice": "all",
        "return_as": "short",
        "segment_length_thresh": segment_length_thresh,
        "spacing": int(spacing),
        "collapse_soma": collapse_soma,
        "n_parallel": int(current_app.config.get("N_PARALLEL")),
        "invalidation_d": int(current_app.config.get("INVALIDATION_D")),
        "selection_point": split_loc,
        "downstream": downstream,
        "root_id_from_point": root_id_from_point,
        "auth_token_key": current_app.config.get("AUTH_TOKEN_KEY"),
        "l2cache": current_app.config.get("USE_L2CACHE", False),
        "target_length": target_length,
        "contrast_lookup": current_app.config.get("CONTRAST_LOOKUP", {}),
        "cv_use_https": current_app.config.get("CV_USE_HTTPS", True),
    }
    print(kwargs)
    job = q.enqueue_call(
        generate_lvl2_paths,
        kwargs=kwargs,
        result_ttl=5000,
        timeout=600,
        retry=Retry(max=2, interval=10),
    )
    return redirect(url_for(".show_result_paths", job_key=job.get_id()))


@bp.route(f"{api_prefix}/datastack/<datastack>/coverpaths", methods=["GET", "POST"])
@auth_required
def lvl2_path_form(datastack):
    if not current_app.config.get("SHOW_PATH_TOOL", False):
        return error_page("Path tool is not enabled")

    form = Lvl2PathForm()
    if form.validate_on_submit():
        root_id = form.root_id.data
        if len(root_id) == 0:
            root_id = None
        segmentation_fallback = form.segmentation_fallback.data

        try:
            root_loc_formatted = encode_root_location(form.root_location.data)
        except Exception as e:
            return error_page(e)
        root_is_soma = form.root_is_soma.data

        split_location_formatted = encode_root_location(form.split_location.data)
        if form.split_option.data == "downstream":
            downstream = True
        else:
            downstream = False
        root_id_from_point = form.root_id_from_root_loc.data

        target_length = form.target_dist.data
        if target_length is not None:
            target_length = int(1_000_000 * target_length)

        num_paths_raw = form.num_paths.data
        num_path_dict = {"all": "all", "five": 5, "ten": 10, "fifteen": 15}
        num_paths = num_path_dict.get(num_paths_raw, "all")

        exclude_short = form.exclude_short.data

        url = url_for(
            ".generate_guidebook_paths",
            datastack=datastack,
            root_id=root_id,
            root_location=root_loc_formatted,
            collapse_soma=root_is_soma,
            num_paths=num_paths,
            exclude_short=exclude_short,
            segmentation_fallback=segmentation_fallback,
            split_location=split_location_formatted,
            downstream=downstream,
            root_id_from_point=root_id_from_point,
            target_length=target_length,
        )
        return redirect(url)

    return render_template(
        "lvl2_paths.html",
        title="Neuron Guidebook",
        form=form,
        version=__version__,
        allow_segmentation=current_app.config.get("ALLOW_SEGMENTATION", False),
    )
