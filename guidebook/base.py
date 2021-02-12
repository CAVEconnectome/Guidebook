import time

import numpy as np
import pandas as pd
import pcg_skel
from annotationframeworkclient import FrameworkClient
from nglui import statebuilder as sb
from scipy import sparse

SK_KWARGS = dict(
    invalidation_d=8000,
    collapse_function="sphere",
    soma_radius=8000,
    compute_radius=False,
    shape_function="single",
)

CONTRAST_LOOKUP = {
    'minnie65_phase3_v1': {'black': 0.35, 'white': 0.7},
}

EP_PROOFREADING_TAGS = ["checked", "error", "correct"]
BP_PROOFREADING_TAGS = ["checked", "error"]


def base_sb_data(
    client, oid, focus_loc=None, black=None, white=None, view_kws={}
):
    if black is None:
        black = CONTRAST_LOOKUP.get(
            client.datastack_name, dict()).get('black', 0)
    if white is None:
        white = CONTRAST_LOOKUP.get(
            client.datastack_name, dict()).get('white', 0)

    state_server = client.state.state_service_endpoint
    img = sb.ImageLayerConfig(
        client.info.image_source(), contrast_controls=True, black=black, white=white
    )
    seg = sb.SegmentationLayerConfig(
        client.info.segmentation_source(),
        fixed_ids=[oid],
        view_kws={"alpha_selected": 0.2, "alpha_3d": 0.6},
    )
    view_kws = {"layout": "3d"}
    view_kws["position"] = focus_loc
    sb_base = sb.StateBuilder(
        layers=[img, seg], state_server=state_server, view_kws=view_kws
    )
    return sb_base, None


def branch_sb_data(
    skf,
    labels=None,
    tags=[],
    set_position=False,
    active=False,
    color="#299bff",
):
    points_bp = sb.PointMapper(
        point_column="bp_locs", group_column="bp_group", set_position=set_position
    )
    bp_layer = sb.AnnotationLayerConfig(
        "branch_points", mapping_rules=points_bp, active=active, color=color, tags=tags
    )
    sb_bp = sb.StateBuilder(layers=[bp_layer], view_kws={"layout": "3d"})

    bps = skf.branch_points_undirected
    if labels is None:
        labels = np.ones(len(skf.vertices))
    bp_lbls = labels[bps]

    bp_df = pd.DataFrame(
        {
            "bps": bps,
            "bp_locs": (skf.vertices[bps] / np.array([4, 4, 40])).tolist(),
            "dfr": skf.distance_to_root[bps],
            "bp_group": bp_lbls,
        }
    )

    return sb_bp, bp_df.sort_values(by=["bp_group", "dfr"])


def end_point_sb_data(skf, labels, tags=[], active=False, color="#FFFFFF"):
    points = sb.PointMapper(point_column="ep_locs", set_position=active)
    ep_layer = sb.AnnotationLayerConfig(
        "end_points", mapping_rules=points, active=active, color=color, tags=tags
    )
    sb_ep = sb.StateBuilder(layers=[ep_layer])

    eps = skf.end_points_undirected
    ep_lbls = labels[eps]

    ep_df = pd.DataFrame(
        {
            "eps": eps,
            "ep_locs": (skf.vertices[eps] / np.array([4, 4, 40])).tolist(),
            "dfr": skf.distance_to_root[eps],
            "ep_group": ep_lbls,
        }
    )

    return sb_ep, ep_df.sort_values(by=["ep_group", "dfr"])


def process_node_groups(skf, cp_max_thresh=200_000):
    cps = skf.cover_paths
    cp_lens = [skf.path_length(cp) for cp in cps]

    cp_ends = np.array([cp[-1] for cp in cps])

    cp_df = pd.DataFrame({"cps": cps, "pathlen": cp_lens, "ends": cp_ends})
    cp_df["root_parent"] = skf.parent_nodes(cp_df["ends"])

    clip = cp_df["pathlen"] > cp_max_thresh
    cp_df[clip].query("root_parent != -1")
    clip_points = cp_df[clip].query("root_parent != -1")["ends"]
    extra_clip_points = skf.child_nodes(skf.root)
    all_clip_pts = np.unique(np.concatenate([clip_points, extra_clip_points]))
    cgph = skf.cut_graph(all_clip_pts)

    _, lbls = sparse.csgraph.connected_components(cgph)
    min_dist_label = [np.min(skf.distance_to_root[lbls == l])
                      for l in np.unique(lbls)]
    labels_ordered = np.unique(lbls)[np.argsort(min_dist_label)]
    new_lbls = np.argsort(labels_ordered)[lbls]
    return new_lbls


def root_sb_data(
    sk, set_position=False, active=False, layer_name="root", voxel_resolution=[4, 4, 40]
):
    pt = sb.PointMapper(point_column="pt", set_position=set_position)
    root_layer = sb.AnnotationLayerConfig(
        layer_name, color="#bfae00", mapping_rules=[pt], active=active
    )
    root_sb = sb.StateBuilder([root_layer])
    root_df = pd.DataFrame(
        {
            "pt": (
                np.atleast_2d(sk.vertices[sk.root]) /
                np.array(voxel_resolution)
            ).tolist(),
        }
    )
    return root_sb, root_df


def generate_lvl2_proofreading(
    datastack,
    root_id,
    root_point=None,
    root_point_resolution=[4, 4, 40],
    refine_branch_points=True,
    refine_end_points=True,
    point_radius=300,
    invalidation_d=3,
    collapse_soma=True,
    return_as="url",
    verbose=True,
):
    if verbose:
        t0 = time.time()
    client = FrameworkClient(datastack)

    if refine_end_points and refine_branch_points:
        refine = 'bpep'
    elif refine_end_points is False:
        refine = 'bp'
    else:
        refine = 'ep'

    l2_sk = pcg_skel.pcg_skeleton(root_id,
                                  client=client,
                                  refine=refine,
                                  root_point=root_point,
                                  root_point_resolution=root_point_resolution,
                                  collapse_soma=collapse_soma,
                                  collapse_radius=10_000,
                                  nan_rounds=None,
                                  invalidation_d=invalidation_d,
                                  root_point_search_radius=point_radius,
                                  )

    sbs = []
    dfs = []

    base_sb, base_df = base_sb_data(
        client, root_id, focus_loc=root_point,
    )
    sbs.append(base_sb)
    dfs.append(base_df)

    rt_sb, rt_df = root_sb_data(l2_sk, set_position=True)
    sbs.append(rt_sb)
    dfs.append(rt_df)

    lbls = process_node_groups(l2_sk, cp_max_thresh=200_000)
    if refine_branch_points:
        bp_sb, bp_df = branch_sb_data(
            l2_sk,
            labels=lbls,
            tags=BP_PROOFREADING_TAGS,
            set_position=False,
            active=True,
        )
        sbs.append(bp_sb)
        dfs.append(bp_df)

    if refine_end_points:
        ep_sb, ep_df = end_point_sb_data(
            l2_sk,
            labels=np.zeros(l2_sk.n_vertices),
            tags=EP_PROOFREADING_TAGS,
            active=False,
            color="#FFFFFF",
        )
        sbs.append(ep_sb)
        dfs.append(ep_df)

    sb_pf = sb.ChainedStateBuilder(sbs)
    if verbose:
        print("\nComplete time: ", time.time() - t0)
    return sb_pf.render_state(
        dfs, return_as=return_as, url_prefix=client.info.viewer_site()
    )
