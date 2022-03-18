import numpy as np
import pandas as pd
from scipy import sparse
from nglui import statebuilder as sb


def base_sb_data(
    client,
    oid,
    focus_loc=None,
    black=None,
    white=None,
    voxel_resolution=[1, 1, 1],
    view_kws={},
    contrast_lookup={},
):
    if black is None:
        black = contrast_lookup.get(client.datastack_name, dict()).get("black", 0)
    if white is None:
        white = contrast_lookup.get(client.datastack_name, dict()).get("white", 1)

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
        layers=[img, seg],
        state_server=state_server,
        view_kws=view_kws,
        resolution=voxel_resolution,
    )
    return sb_base, None


def branch_sb_data(
    skf,
    labels=None,
    tags=[],
    set_position=False,
    active=False,
    color="#299bff",
    voxel_resolution=[1, 1, 1],
):
    points_bp = sb.PointMapper(
        point_column="bp_locs", group_column="bp_group", set_position=set_position
    )
    bp_layer = sb.AnnotationLayerConfig(
        "branch_points", mapping_rules=points_bp, active=active, color=color, tags=tags
    )
    sb_bp = sb.StateBuilder(
        layers=[bp_layer],
        view_kws={"layout": "3d"},
        resolution=voxel_resolution,
    )

    bps = skf.branch_points_undirected
    if labels is None:
        labels = np.ones(len(skf.vertices))
    bp_lbls = labels[bps]

    bp_df = pd.DataFrame(
        {
            "bps": bps,
            "bp_locs": (skf.vertices[bps] / np.array(voxel_resolution)).tolist(),
            "dfr": skf.distance_to_root[bps],
            "bp_group": bp_lbls,
        }
    )

    return sb_bp, bp_df.sort_values(by=["bp_group", "dfr"])


def end_point_sb_data(
    skf,
    labels,
    tags=[],
    active=False,
    color="#FFFFFF",
    voxel_resolution=[1, 1, 1],
    omit_indices=[],
):
    points = sb.PointMapper(point_column="ep_locs", set_position=active)
    ep_layer = sb.AnnotationLayerConfig(
        "end_points", mapping_rules=points, active=active, color=color, tags=tags
    )
    sb_ep = sb.StateBuilder(layers=[ep_layer], resolution=voxel_resolution)

    eps = skf.end_points_undirected
    eps = eps[~np.isin(eps, omit_indices)]
    ep_lbls = labels[eps]

    ep_df = pd.DataFrame(
        {
            "eps": eps,
            "ep_locs": (skf.vertices[eps] / np.array(voxel_resolution)).tolist(),
            "dfr": skf.distance_to_root[eps],
            "ep_group": ep_lbls,
        }
    )

    return sb_ep, ep_df.sort_values(by=["ep_group", "dfr"])


def selection_point_sb_data(
    selection_point,
    direction,
    active=False,
    color="#FF2200",
    voxel_resolution=[1, 1, 1],
):
    points = sb.PointMapper(point_column="pt_locs", set_position=active)
    sp_layer = sb.AnnotationLayerConfig(
        f"{direction}_point",
        mapping_rules=points,
        active=active,
        color=color,
    )
    sb_sp = sb.StateBuilder(layers=[sp_layer], resolution=voxel_resolution)
    pts = np.atleast_2d(selection_point)
    sp_df = pd.DataFrame(
        {
            "pt_locs": pts.tolist(),
        }
    )
    return sb_sp, sp_df


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
    min_dist_label = [np.min(skf.distance_to_root[lbls == l]) for l in np.unique(lbls)]
    labels_ordered = np.unique(lbls)[np.argsort(min_dist_label)]
    new_lbls = np.argsort(labels_ordered)[lbls]
    return new_lbls


def root_sb_data(
    sk,
    set_position=False,
    active=False,
    layer_name="root",
    voxel_resolution=[1, 1, 1],
):
    pt = sb.PointMapper(point_column="pt", set_position=set_position)
    root_layer = sb.AnnotationLayerConfig(
        layer_name, color="#bfae00", mapping_rules=[pt], active=active
    )
    root_point = sk._rooted.vertices[
        sk._rooted.root
    ]  # Works even if root is not in the skeleton mask
    root_sb = sb.StateBuilder([root_layer], resolution=voxel_resolution)
    root_df = pd.DataFrame(
        {
            "pt": (np.atleast_2d(root_point) / np.array(voxel_resolution)).tolist(),
        }
    )
    return root_sb, root_df


def topo_point_construction(
    l2_sk,
    l2dict,
    root_id,
    root_point,
    root_point_resolution,
    refine_branch_points,
    refine_end_points,
    selection_point,
    selection_skinds,
    downstream,
    client,
    ep_proofreading_tags=[],
    bp_proofreading_tags=[],
    contrast_lookup={},
):
    sbs = []
    dfs = []

    base_sb, base_df = base_sb_data(
        client,
        root_id,
        focus_loc=root_point,
        contrast_lookup=contrast_lookup,
    )
    sbs.append(base_sb)
    dfs.append(base_df)

    rt_sb, rt_df = root_sb_data(
        l2_sk,
        set_position=True,
        voxel_resolution=root_point_resolution,
    )
    sbs.append(rt_sb)
    dfs.append(rt_df)

    lbls = process_node_groups(l2_sk, cp_max_thresh=250_000)
    if refine_branch_points:
        bp_sb, bp_df = branch_sb_data(
            l2_sk,
            labels=lbls,
            tags=bp_proofreading_tags,
            set_position=False,
            active=True,
            voxel_resolution=root_point_resolution,
        )
        sbs.append(bp_sb)
        dfs.append(bp_df)

    if refine_end_points:
        ep_sb, ep_df = end_point_sb_data(
            l2_sk,
            labels=np.zeros(l2_sk.n_vertices),
            tags=ep_proofreading_tags,
            active=False,
            color="#FFFFFF",
            voxel_resolution=root_point_resolution,
            omit_indices=selection_skinds,
        )
        sbs.append(ep_sb)
        dfs.append(ep_df)

    if len(selection_skinds) > 0:
        direction = {True: "downstream", False: "upstream"}
        sp_sb, sp_df = selection_point_sb_data(
            selection_point,
            direction.get(downstream),
            voxel_resolution=root_point_resolution,
        )
        sbs.append(sp_sb)
        dfs.append(sp_df)

    return sbs, dfs