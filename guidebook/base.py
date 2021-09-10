import time
import pcg_skel
import numpy as np
from caveclient import CAVEclient
from nglui import statebuilder as sb
from pcg_skel.chunk_tools import get_root_id_from_point, get_closest_lvl2_chunk
from .topo_points import topo_point_construction
from .cover_review import construct_cover_paths

from .parameters import (
    GUIDEBOOK_EXPECTED_RESOLUTION,
    PATH_SPACING,
)


def mask_skeleton(
    root_id, sk, l2dict, selection_point, downstream, client, voxel_resolution, radius
):
    """Mask the skeleton up or downstream of selection point"""
    selection_l2id = get_closest_lvl2_chunk(
        selection_point,
        root_id,
        client,
        voxel_resolution=voxel_resolution,
        radius=radius,
    )

    selection_skid = l2dict[selection_l2id]

    mask = np.full(sk.n_vertices, False)
    ds_skinds = sk.downstream_nodes(selection_skid)
    mask[ds_skinds] = True

    if downstream is False:
        mask = np.invert(mask)
        mask[selection_skid] = ~mask[selection_skid]

    return sk.apply_mask(mask), selection_l2id


def generate_lvl2_paths(
    datastack,
    root_id,
    server_address,
    root_point=None,
    root_point_resolution=GUIDEBOOK_EXPECTED_RESOLUTION,
    n_choice="all",
    segment_length_thresh=0,
    spacing=PATH_SPACING,
    interp_method="linear",
    selection_point=None,
    downstream=True,
    invalidation_d=3,
    collapse_soma=True,
    n_parallel=1,
    point_radius=300,
    root_id_from_point=False,
    auth_token_key=None,
    return_as="url",
    verbose=True,
):
    if verbose:
        t0 = time.time()

    client = CAVEclient(
        datastack, server_address=server_address, auth_token_key=auth_token_key
    )

    if root_id_from_point and root_id is None:
        root_id = get_root_id_from_point(root_point, root_point_resolution, client)
        if root_id == 0:
            raise ValueError("Root point was not on any segmentation")

    l2_sk, (l2dict, l2dict_r) = pcg_skel.pcg_skeleton(
        root_id,
        client=client,
        refine="all",
        root_point=root_point,
        root_point_resolution=root_point_resolution,
        collapse_soma=collapse_soma,
        collapse_radius=10_000,
        nan_rounds=20,
        return_l2dict=True,
        invalidation_d=invalidation_d,
        root_point_search_radius=point_radius,
        segmentation_fallback=False,
        n_parallel=n_parallel,
    )

    if selection_point is not None:
        l2_sk, _ = mask_skeleton(
            root_id,
            l2_sk,
            l2dict,
            selection_point=selection_point,
            downstream=downstream,
            client=client,
            voxel_resolution=root_point_resolution,
            radius=point_radius,
        )

    dfs, sbs = construct_cover_paths(
        l2_sk,
        n_choice,
        spacing,
        root_id,
        segment_length_thresh,
        client,
        root_point_resolution,
        interp_method,
        selection_point,
    )

    csb = sb.ChainedStateBuilder(sbs)
    if verbose:
        print("\nComplete time: ", time.time() - t0)

    return csb.render_state(
        dfs, return_as=return_as, url_prefix=client.info.viewer_site()
    )


def generate_lvl2_proofreading(
    datastack,
    root_id,
    server_address,
    root_point=None,
    root_point_resolution=GUIDEBOOK_EXPECTED_RESOLUTION,
    refine_branch_points=True,
    refine_end_points=True,
    point_radius=300,
    invalidation_d=3,
    collapse_soma=True,
    return_as="url",
    verbose=True,
    segmentation_fallback=True,
    selection_point=None,
    downstream=True,
    n_parallel=1,
    root_id_from_point=False,
    auth_token_key=None,
):
    if verbose:
        t0 = time.time()
    client = CAVEclient(
        datastack, server_address=server_address, auth_token_key=auth_token_key
    )

    if refine_end_points and refine_branch_points:
        refine = "bpep"
    elif refine_end_points is False:
        refine = "bp"
    else:
        refine = "ep"

    if root_id_from_point and root_id is None:
        root_id = get_root_id_from_point(root_point, root_point_resolution, client)
        if root_id == 0:
            raise ValueError("Root point was not on any segmentation")

    l2_sk, (l2dict, l2dict_r) = pcg_skel.pcg_skeleton(
        root_id,
        client=client,
        refine=refine,
        root_point=root_point,
        root_point_resolution=root_point_resolution,
        collapse_soma=collapse_soma,
        collapse_radius=10_000,
        nan_rounds=None,
        return_l2dict=True,
        invalidation_d=invalidation_d,
        root_point_search_radius=point_radius,
        segmentation_fallback=segmentation_fallback,
        n_parallel=n_parallel,
    )

    if selection_point is not None:
        l2_sk, selection_l2id = mask_skeleton(
            root_id,
            l2_sk,
            l2dict,
            selection_point=selection_point,
            downstream=downstream,
            client=client,
            voxel_resolution=root_point_resolution,
            radius=point_radius,
        )
        selection_skinds = l2_sk.filter_unmasked_indices(
            np.array([l2dict[selection_l2id]])
        )
    else:
        selection_skinds = []

    sbs, dfs = topo_point_construction(
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
    )

    sb_pf = sb.ChainedStateBuilder(sbs)
    if verbose:
        print("\nComplete time: ", time.time() - t0)
    return sb_pf.render_state(
        dfs, return_as=return_as, url_prefix=client.info.viewer_site()
    )
