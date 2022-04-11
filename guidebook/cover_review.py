import pandas as pd
import numpy as np
from scipy import interpolate
from nglui import statebuilder as sb
from .topo_points import selection_point_sb_data


def sample_end_points(sk, n_choice=None, target_length=None, ep_segment_thresh=0):
    if target_length is not None:
        return end_points_for_target_length(sk, target_length, ep_segment_thresh)
    if n_choice is not None:
        return end_points_random_choice(sk, n_choice, ep_segment_thresh)
    else:
        raise ValueError("Either n_choice or target_length must be specified")


def end_points_random_choice(sk, n_choice, ep_segment_thresh):
    eps = viable_end_points(sk, ep_segment_thresh=int(ep_segment_thresh))

    if n_choice == "all":
        return eps
    else:
        return np.random.choice(eps, int(n_choice), replace=False)


def end_points_for_target_length(sk, target_length, ep_segment_thresh):
    eps = viable_end_points(sk, ep_segment_thresh=int(ep_segment_thresh))

    selected_ep = np.array([], dtype=int)
    net_path_length = 0
    while net_path_length < target_length and len(selected_ep) < len(eps):
        add_ep = np.random.choice(eps[~np.isin(eps, selected_ep)])
        selected_ep = np.append(selected_ep, add_ep)
        cps = sk.cover_paths_specific(selected_ep)
        net_path_length = sum([sk.path_length(p) for p in cps])
    return selected_ep


def viable_end_points(sk, ep_segment_thresh=0):
    """Determines which end points are good for sampling by allowing ones on short segments to be"""
    eps = []
    cp_len = []
    for cp in sk.cover_paths:
        eps.append(cp[0])
        cp_len.append(sk.path_length(cp))
    eps = np.array(eps)
    cp_len = np.array(cp_len)
    return eps[cp_len > ep_segment_thresh]


def interpolate_path(
    path, sk, spacing=2500, interp_method="linear", voxel_resolution=[4, 4, 40]
):
    path_verts = sk.vertices[path] / voxel_resolution
    ds = sk.distance_to_root[path[0]] - sk.distance_to_root[path]

    interp_x = interpolate.interp1d(
        ds,
        path_verts[:, 0],
        kind=interp_method,
    )
    interp_y = interpolate.interp1d(
        ds,
        path_verts[:, 1],
        kind=interp_method,
    )
    interp_z = interpolate.interp1d(
        ds,
        path_verts[:, 2],
        kind=interp_method,
    )

    vals = np.arange(0, max(ds), spacing)
    path_pts = np.vstack(
        [
            interp_x(vals),
            interp_y(vals),
            interp_z(vals),
        ]
    ).T
    return path_pts


def generate_path_df(
    paths,
    sk,
    interp_spacing=2_000,
    interp_method="linear",
    voxel_resolution=[4, 4, 40],
):
    grp = []
    ptsA = []
    ptsB = []

    grp_ind = 0
    for path in paths:
        path_interp = interpolate_path(
            path,
            sk,
            spacing=interp_spacing,
            interp_method=interp_method,
            voxel_resolution=voxel_resolution,
        )
        path_interp = path_interp[::-1]

        ptsA.append(path_interp[:-1])
        ptsB.append(path_interp[1:])

        grp.append(grp_ind * np.ones(len(path_interp) - 1))
        grp_ind += 1

    ptsA = np.concatenate(ptsA).tolist()
    ptsB = np.concatenate(ptsB).tolist()
    grp = np.concatenate(grp)
    return pd.DataFrame(
        {
            "ptA": ptsA,
            "ptB": ptsB,
            "group": grp,
        }
    )


def base_builder(client, root_id, voxel_resolution, contrast_lookup={}):
    black = contrast_lookup.get(client.datastack_name, dict()).get("black", 0)
    white = contrast_lookup.get(client.datastack_name, dict()).get("white", 1)
    img = sb.ImageLayerConfig(
        client.info.image_source(), contrast_controls=True, black=black, white=white
    )
    seg = sb.SegmentationLayerConfig(
        client.info.segmentation_source(),
        fixed_ids=[root_id],
        alpha_3d=0.6,
    )
    sb_base = sb.StateBuilder(
        layers=[img, seg],
        state_server=client.state.state_service_endpoint,
        resolution=voxel_resolution,
    )
    return sb_base, None


def path_layers(l2_sk, paths, spacing, interp_method, voxel_resolution):

    df = generate_path_df(
        paths,
        l2_sk,
        spacing,
        interp_method=interp_method,
        voxel_resolution=voxel_resolution,
    )

    anno = sb.AnnotationLayerConfig(
        "selected_paths",
        mapping_rules=sb.LineMapper(
            "ptA",
            "ptB",
            group_column="group",
            set_position=True,
        ),
    )

    return sb.StateBuilder(layers=[anno], resolution=voxel_resolution), df


def construct_cover_paths(
    l2_sk,
    n_choice,
    spacing,
    root_id,
    ep_segment_thresh,
    client,
    target_length=None,
    voxel_resolution=[1, 1, 1],
    interp_method="linear",
    selection_point=None,
    downstream=True,
    contrast_lookup={},
):
    sbs = []
    dfs = []

    sb_base, df_base = base_builder(
        client, root_id, voxel_resolution, contrast_lookup=contrast_lookup
    )
    sbs.append(sb_base)
    dfs.append(df_base)

    if selection_point is not None:
        direction = {True: "downstream", False: "upstream"}
        sb_sp, sp_df = selection_point_sb_data(
            selection_point=selection_point,
            direction=direction.get(downstream),
            voxel_resolution=voxel_resolution,
        )
        sbs.append(sb_sp)
        dfs.append(sp_df)

    eps = sample_end_points(
        l2_sk,
        n_choice=n_choice,
        target_length=target_length,
        ep_segment_thresh=ep_segment_thresh,
    )

    paths = l2_sk.cover_paths_specific(eps, include_parent=True)

    sb_path, df_path = path_layers(
        l2_sk, paths, spacing, interp_method, voxel_resolution
    )
    sbs.append(sb_path)
    dfs.append(df_path)

    return dfs, sbs
