import time
import pcg_skel
import numpy as np
from caveclient import CAVEclient
from nglui import statebuilder as sb
from pcg_skel.chunk_tools import get_root_id_from_point, get_closest_lvl2_chunk
from .topo_points import topo_point_construction
from .cover_review import construct_cover_paths


def link_shortened_state(state, client):
    state_id = client.state.upload_state_json(state)
    return client.state.build_neuroglancer_url(state_id)


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
    root_point_resolution=[1, 1, 1],
    n_choice="all",
    segment_length_thresh=0,
    spacing=2_000,
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
    l2cache=False,
    target_length=None,
    contrast_lookup={},
    cv_use_https=True,
):
    if verbose:
        t0 = time.time()

    client = CAVEclient(
        datastack, server_address=server_address, auth_token_key=auth_token_key
    )

    # If not using https, also use local secrets not client secrets in guidebook to avoid permissions errors
    cv = client.info.segmentation_cloudvolume(
        use_client_secret=cv_use_https, use_https=cv_use_https
    )

    if root_id_from_point and root_id is None:
        root_id = get_root_id_from_point(root_point, root_point_resolution, client)
        if root_id == 0:
            raise ValueError("Root point was not on any segmentation")

    l2_sk, (l2dict, l2dict_r) = pcg_skel.pcg_skeleton(
        root_id,
        client=client,
        cv=cv,
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
        n_parallel=1,
        l2cache=l2cache,
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
        l2_sk=l2_sk,
        n_choice=n_choice,
        spacing=spacing,
        root_id=root_id,
        ep_segment_thresh=segment_length_thresh,
        client=client,
        voxel_resolution=root_point_resolution,
        interp_method=interp_method,
        selection_point=selection_point,
        contrast_lookup=contrast_lookup,
        target_length=target_length,
    )

    csb = sb.ChainedStateBuilder(sbs)
    if verbose:
        print("\nComplete time: ", time.time() - t0)

    if return_as == "short":
        state = csb.render_state(
            dfs, return_as="dict", url_prefix=client.info.viewer_site()
        )
        return link_shortened_state(state, client)
    else:
        return csb.render_state(
            dfs, return_as=return_as, url_prefix=client.info.viewer_site()
        )


def generate_lvl2_proofreading(
    datastack,
    root_id,
    server_address,
    root_point=None,
    root_point_resolution=[1, 1, 1],
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
    l2cache=False,
    contrast_lookup={},
    ep_tags=[],
    bp_tags=[],
    cv_use_https=True,
):
    if verbose:
        t0 = time.time()
    client = CAVEclient(
        datastack, server_address=server_address, auth_token_key=auth_token_key
    )
    cv = client.info.segmentation_cloudvolume(
        use_client_secret=cv_use_https, use_https=cv_use_https
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
        cv=cv,
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
        l2cache=l2cache,
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
        contrast_lookup=contrast_lookup,
        ep_proofreading_tags=ep_tags,
        bp_proofreading_tags=bp_tags,
    )

    if verbose:
        print("\nComplete time: ", time.time() - t0)

    sb_pf = sb.ChainedStateBuilder(sbs)

    if return_as == "short":
        state = sb_pf.render_state(
            dfs, return_as="dict", url_prefix=client.info.viewer_site()
        )
        return link_shortened_state(state, client)
    else:
        return sb_pf.render_state(
            dfs, return_as=return_as, url_prefix=client.info.viewer_site()
        )