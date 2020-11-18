import pandas as pd
import numpy as np
from annotationframeworkclient import FrameworkClient
from scipy import sparse, spatial
import fastremap
import cloudvolume
import requests
from meshparty import trimesh_io, skeleton, skeletonize, mesh_filters
from nglui import statebuilder as sb
import time

MIN_CC_SIZE = 1000
SK_KWARGS = dict(invalidation_d=8000,
                 collapse_function='sphere',
                 soma_radius=8000,
                 compute_radius=False,
                 shape_function='single')

EP_PROOFREADING_TAGS = []
BP_PROOFREADING_TAGS = ['checked', 'error']


def chunk_dims(cv):
    """Get the size of a single chunk"""
    dims = chunk_to_nm([1, 1, 1], cv)-chunk_to_nm([0, 0, 0], cv)
    return np.squeeze(dims)


def process_mesh(oid, client, mm, min_component_size=1000):
    """Download mesh and postprocess for link edges and dust"""
    mesh = mm.mesh(seg_id=oid)
    mesh.add_link_edges(seg_id=oid, client=client.chunkedgraph)
    mf = mesh_filters.filter_components_by_size(mesh, min_size=1000)
    meshf = mesh.apply_mask(mf)
    return meshf


def process_skeleton_from_mesh(mesh, root_loc=None, root_is_soma=False, close_mean=10000, sk_kwargs={}):
    if root_is_soma:
        soma_loc = root_loc
    else:
        soma_loc = None
    sk = skeletonize.skeletonize_mesh(mesh,
                                      soma_pt=soma_loc,
                                      **sk_kwargs)

    if not root_is_soma and root_loc is not None:
        _, skind = sk.kdtree.query(root_loc)
        sk.reroot(skind)

    _, lbls = sparse.csgraph.connected_components(sk.csgraph)
    _, cnt = np.unique(lbls, return_counts=True)

    if soma_loc is not None:
        modest_ccs = np.flatnonzero(cnt < MIN_CC_SIZE)
        remove_ccs = []
        for cc in modest_ccs:
            if np.mean(np.linalg.norm(sk.vertices[lbls == cc] - soma_loc, axis=1)) < close_mean:
                remove_ccs.append(cc)

        keep_mask = ~np.isin(lbls, remove_ccs)
        skout = sk.apply_mask(keep_mask)
    else:
        skout = sk
    return skout


def base_sb_data(client, oid, focus_loc=None, fixed_ids=[], black=0.35, white=0.7, view_kws={}):
    state_server = client.state.state_service_endpoint
    img = sb.ImageLayerConfig(client.info.image_source(),
                              contrast_controls=True,
                              black=black, white=white)
    seg = sb.SegmentationLayerConfig(client.info.segmentation_source(),
                                     fixed_ids=[oid],
                                     view_kws={'alpha_selected': 0.2,
                                               'alpha_3d': 0.6})
    view_kws = {'layout': '3d'}
    view_kws['position'] = focus_loc
    sb_base = sb.StateBuilder(layers=[img, seg],
                              state_server=state_server,
                              view_kws=view_kws)
    return sb_base, None


def branch_sb_data(client, oid, skf, labels=None, tags=[], set_position=False, active=False, color='#299bff'):
    points_bp = sb.PointMapper(
        point_column='bp_locs', group_column='bp_group', set_position=set_position)
    bp_layer = sb.AnnotationLayerConfig(
        'branch_points', mapping_rules=points_bp, active=active, color=color, tags=tags)
    sb_bp = sb.StateBuilder(layers=[bp_layer], view_kws={'layout': '3d'})

    bps = skf.branch_points_undirected
    if labels is None:
        labels = np.ones(len(skf.vertices))
    bp_lbls = labels[bps]

    bp_df = pd.DataFrame({'bps': bps,
                          'bp_locs': (skf.vertices[bps] / np.array([4, 4, 40])).tolist(),
                          'dfr': skf.distance_to_root[bps],
                          'bp_group': bp_lbls})

    return sb_bp, bp_df.sort_values(by=['bp_group', 'dfr'])


def end_point_sb_data(client, oid, skf, labels, tags=[], active=False, color='#299bff'):
    points = sb.PointMapper(
        point_column='ep_locs', group_column='ep_group', set_position=active)
    ep_layer = sb.AnnotationLayerConfig(
        'branch_points', mapping_rules=points, active=active, color=color, tags=tags)
    sb_ep = sb.StateBuilder(layers=[ep_layer])

    eps = skf.end_points_undirected
    ep_lbls = labels[eps]

    ep_df = pd.DataFrame({'eps': eps,
                          'ep_locs': (skf.vertices[eps] / np.array([4, 4, 40])).tolist(),
                          'dfr': skf.distance_to_root[eps],
                          'ep_group': ep_lbls})

    return sb_ep, ep_df.sort_values(by=['ep_group', 'dfr'])


def proofer_statebuilder(oid, client, root_pt=None, bp_proofreading_tags=[], ep_proofreading_tags=[], bp_active=False):

    state_server = client.state.state_service_endpoint
    img = sb.ImageLayerConfig(client.info.image_source(),
                              contrast_controls=True,
                              black=0.35, white=0.7)
    seg = sb.SegmentationLayerConfig(client.info.segmentation_source(),
                                     fixed_ids=[oid],
                                     view_kws={'alpha_selected': 0.2, 'alpha_3d': 0.6})

    points_bp = sb.PointMapper(
        point_column='bp_locs', group_column='bp_group', set_position=bp_active)
    bp_layer = sb.AnnotationLayerConfig(
        'branch_points', mapping_rules=points_bp,
        active=True, color='#299bff', tags=bp_proofreading_tags)

    points_ep = sb.PointMapper(point_column='ep_locs')
    ep_layer = sb.AnnotationLayerConfig(
        'end_points', mapping_rules=points_ep,
        color='#ffffff', tags=ep_proofreading_tags)

    if root_pt is not None:
        view_kws = {'position': root_pt}
    else:
        view_kws = {}

    sb_ep = sb.StateBuilder(layers=[img, seg, ep_layer],
                            state_server=state_server, view_kws=view_kws)
    sb_bp = sb.StateBuilder(layers=[bp_layer], view_kws={'layout': '3d'})

    sb_all = sb.ChainedStateBuilder([sb_ep, sb_bp])

    return sb_all


def process_node_groups(skf, cp_max_thresh=200_000):
    cps = skf.cover_paths
    cp_lens = [skf.path_length(cp) for cp in cps]

    cp_ends = np.array([cp[-1] for cp in cps])

    cp_df = pd.DataFrame({'cps': cps, 'pathlen': cp_lens, 'ends': cp_ends})
    cp_df['root_parent'] = skf.parent_nodes(cp_df['ends'])

    clip = cp_df['pathlen'] > cp_max_thresh
    cp_df[clip].query('root_parent != -1')
    clip_points = cp_df[clip].query('root_parent != -1')['ends']
    extra_clip_points = skf.child_nodes(skf.root)
    all_clip_pts = np.unique(np.concatenate([clip_points, extra_clip_points]))
    cgph = skf.cut_graph(all_clip_pts)

    _, lbls = sparse.csgraph.connected_components(cgph)
    min_dist_label = [np.min(skf.distance_to_root[lbls == l])
                      for l in np.unique(lbls)]
    labels_ordered = np.unique(lbls)[np.argsort(min_dist_label)]
    new_lbls = np.argsort(labels_ordered)[lbls]
    return new_lbls


def build_topo_dataframes(skf, cp_percentile):
    cps = skf.cover_paths
    cp_lens = [skf.path_length(cp) for cp in cps]

    cp_ends = np.array([cp[-1] for cp in cps])

    cp_df = pd.DataFrame({'cps': cps, 'pathlen': cp_lens, 'ends': cp_ends})
    cp_df['root_parent'] = skf.parent_nodes(cp_df['ends'])

    cp_max_thresh = np.percentile(cp_lens, cp_percentile)

    clip = cp_df['pathlen'] > cp_max_thresh
    cp_df[clip].query('root_parent != -1')
    clip_points = cp_df[clip].query('root_parent != -1')['ends']
    extra_clip_points = skf.child_nodes(skf.root)
    all_clip_pts = np.unique(np.concatenate([clip_points, extra_clip_points]))
    cgph = skf.cut_graph(all_clip_pts)

    _, lbls = sparse.csgraph.connected_components(cgph)
    min_dist_label = [np.min(skf.distance_to_root[lbls == l])
                      for l in np.unique(lbls)]
    labels_ordered = np.unique(lbls)[np.argsort(min_dist_label)]
    new_lbls = np.argsort(labels_ordered)[lbls]

    bps = skf.branch_points_undirected
    bp_lbls = new_lbls[bps]

    bp_df = pd.DataFrame({'bps': bps,
                          'bp_locs': (skf.vertices[bps] / np.array([4, 4, 40])).tolist(),
                          'dfr': skf.distance_to_root[bps],
                          'bp_group': bp_lbls})

    eps = skf.end_points_undirected
    ep_lbls = new_lbls[eps]
    ep_df = pd.DataFrame({'eps': eps,
                          'ep_locs': (skf.vertices[eps] / np.array([4, 4, 40])).tolist(),
                          'dfr': skf.distance_to_root[eps],
                          'ep_group': ep_lbls})
    return ep_df, bp_df


def process_points_from_skeleton(oid, sk, client, cp_percentile=98,
                                 bp_proofreading_tags=EP_PROOFREADING_TAGS,
                                 ep_proofreading_tags=BP_PROOFREADING_TAGS):

    ep_df, bp_df = build_topo_dataframes(sk, cp_percentile=cp_percentile)
    sb_dfs = (ep_df.sort_values(by=['ep_group', 'dfr']),
              bp_df.sort_values(by=['bp_group', 'dfr']))
    pf_sb = proofer_statebuilder(
        oid, client, sk.vertices[sk.root] / np.array([4, 4, 40]), bp_proofreading_tags, ep_proofreading_tags)
    return pf_sb, sb_dfs


def generate_proofreading_state(datastack,
                                root_id,
                                branch_points=True,
                                end_points=True,
                                root_loc=None,
                                root_is_soma=False,
                                auth_token=None,
                                return_as='html',
                                min_mesh_component_size=1000,
                                ):
    """Go through the steps to generate a proofreading state from the root id"""
    client = FrameworkClient(datastack, auth_token=auth_token)
    mm = trimesh_io.MeshMeta(
        cache_size=0, cv_path=client.info.segmentation_source())
    mesh = process_mesh(root_id, client, mm,
                        min_component_size=min_mesh_component_size)
    skf = process_skeleton_from_mesh(mesh, root_loc=root_loc,
                                     root_is_soma=root_is_soma, sk_kwargs=SK_KWARGS)
    pf_sb, sb_dfs = process_points_from_skeleton(root_id, skf, client)

    state = pf_sb.render_state(
        sb_dfs, return_as=return_as, url_prefix=client.info.viewer_site())

    return state


######
# Chunkedgraph skeletonization
######

def nm_to_chunk(xyz_nm, cv, voxel_resolution=[4, 4, 40], mip_scaling=[2, 2, 1]):
    x_vox = np.atleast_2d(xyz_nm) / (np.array(mip_scaling)
                                     * np.array(voxel_resolution))
    offset_vox = np.array(cv.mesh.meta.meta.voxel_offset(0))
    return (x_vox + offset_vox) / np.array(cv.mesh.meta.meta.graph_chunk_size)


def chunk_to_nm(xyz_ch, cv, voxel_resolution=[4, 4, 40], mip_scaling=[2, 2, 1]):
    x_vox = np.atleast_2d(xyz_ch) * cv.mesh.meta.meta.graph_chunk_size
    return (x_vox + np.array(cv.mesh.meta.meta.voxel_offset(0))) * voxel_resolution * mip_scaling


def get_lvl2_graph(root_id, client):
    url = f'https://minnie.microns-daf.com/segmentation/api/v1/table/{client.chunkedgraph.table_name}/node/{root_id}/lvl2_graph'
    r = requests.get(url, headers=client.auth.request_header)
    eg = r.json()
    eg_arr = np.array(eg['edge_graph'], dtype=np.int)
    return np.unique(np.sort(eg_arr, axis=1), axis=0)


def build_spatial_graph(lvl2_edge_graph, cv):
    lvl2_ids = np.unique(lvl2_edge_graph)
    l2dict = {l2id: ii for ii, l2id in enumerate(lvl2_ids)}
    eg_arr_rm = fastremap.remap(lvl2_edge_graph, l2dict)
    l2dict_reversed = {ii: l2id for l2id, ii in l2dict.items()}

    x_ch = [np.array(cv.mesh.meta.meta.decode_chunk_position(l))
            for l in lvl2_ids]
    return eg_arr_rm, l2dict, l2dict_reversed, x_ch


def skeletonize_lvl2_graph(mesh_chunk, root_pt=None, cv=None, client=None, voxel_resolution=None, mip_scaling=None, invalidation_d=3, point_radius=200):
    sk_ch = skeletonize.skeletonize_mesh(
        mesh_chunk, invalidation_d=invalidation_d,
        collapse_soma=False, compute_radius=False,
        remove_zero_length_edges=False)
    return sk_ch


def lvl2_branch_fragment_locs(sk_ch, lvl2dict_reversed, cv):
    br_minds = sk_ch.mesh_index[sk_ch.branch_points_undirected]
    branch_l2s = list(map(lambda x: lvl2dict_reversed[x], br_minds))
    l2meshes = cv.mesh.get_meshes_on_bypass(branch_l2s, allow_missing=True)

    l2means = []
    missing_ids = []
    for l2id in branch_l2s:
        try:
            l2m = np.mean(l2meshes[l2id].vertices, axis=0)
            _, ii = spatial.cKDTree(l2meshes[l2id].vertices).query(l2m)
            l2means.append(l2meshes[l2id].vertices[ii])
        except:
            missing_ids.append(l2id)
            l2means.append(np.array([np.nan, np.nan, np.nan]))
    l2means = np.vstack(l2means)
    return l2means, missing_ids


def get_lvl2_skeleton(client, root_id, convert_to_nm=False, refine_branch_points=False,
                      root_point=None, point_radius=200, invalidation_d=3, verbose=False,
                      auto_remesh=False, allow_missing_chunks=True):
    """Get branch points of the level 2 skeleton for a root id.

    Parameters
    ----------
    datastack : str
        Datastack name
    root_id : int
        Root id of object to skeletonize
    invalidation_d : int, optional
        Invalidation distance in chunk increments

    Returns
    -------
    Branch point locations
        Branch point locations in mesh space (nms)
    """
    if verbose:
        import time
        t0 = time.time()

    cv = cloudvolume.CloudVolume(
        client.info.segmentation_source(), use_https=True, progress=False, bounded=False)

    lvl2_eg = get_lvl2_graph(root_id, client)
    if verbose:
        t1 = time.time()
        print('\nTime to return graph: ', t1-t0)

    eg, l2dict, l2dict_reversed, x_ch = build_spatial_graph(lvl2_eg, cv)

    mesh_chunk = trimesh_io.Mesh(vertices=x_ch, faces=[], link_edges=eg)

    if root_point is not None:
        if verbose:
            t2 = time.time()
        lvl2_root_chid, lvl2_root_loc = get_closest_lvl2_chunk(
            root_point, root_id, client=client, cv=cv, radius=point_radius, return_point=True)
        root_mesh_index = l2dict[lvl2_root_chid]
        if verbose:
            print('\n Time to get root index: ', time.time()-t2)
    else:
        root_mesh_index = None
        lvl2_root_loc = None

    if verbose:
        t3 = time.time()
    sk_ch = skeletonize.skeletonize_mesh(
        mesh_chunk, invalidation_d=invalidation_d,
        collapse_soma=False, compute_radius=False,
        root_index=root_mesh_index, remove_zero_length_edges=False)
    if verbose:
        print('\n Time to skeletonize: ', time.time()-t3)

    if refine_branch_points:
        if verbose:
            t4 = time.time()
        sk_ch, missing_ids = refine_skeleton(sk_ch, l2dict_reversed,
                                             cv, convert_to_nm, root_location=lvl2_root_loc)
        if verbose:
            print('\n Time to refine branch points, ', time.time()-t4)
        if len(missing_ids) > 0:
            if auto_remesh:
                # remesh missing and one-hop neighbors
                remesh_ids = np.unique(
                    eg[np.any(np.isin(eg, missing_ids), axis=1)])
                client.chunkedgraph.remesh_level2_chunks(remesh_ids)
                if not allow_missing_chunks:
                    raise ValueError(
                        f'Regenerating mesh for level 2 ids and their neighbors: {missing_ids}. Try again in a few minutes.')
            else:
                if not allow_missing_chunks:
                    raise ValueError(
                        f'No mesh found for level 2 ids: {missing_ids}')

    return sk_ch, l2dict_reversed


def lvl2_statebuilder(datastack, root_id):
    client = FrameworkClient(datastack)
    img = sb.ImageLayerConfig(client.info.image_source(
    ), contrast_controls=True, black=0.35, white=0.66)
    seg = sb.SegmentationLayerConfig(
        client.info.segmentation_source(), fixed_ids=[root_id])

    anno = sb.AnnotationLayerConfig(
        'branch_points', mapping_rules=sb.PointMapper(), array_data=True, active=True)
    bp_statebuilder = sb.StateBuilder(
        [img, seg, anno], url_prefix=client.info.viewer_site())
    return bp_statebuilder


def get_closest_lvl2_chunk(point, root_id, client, cv=None, resolution=[4, 4, 40], mip_rescale=[2, 2, 1], radius=200, return_point=False):
    """Get the closest level 2 chunk on a root id 

    Parameters
    ----------
    point : array-like
        Point in space.
    root_id : int
        Root id of the object
    client : FrameworkClient
        Framework client to access data
    cv : cloudvolume.CloudVolume, optional
        Predefined cloudvolume, generated if None. By default None
    resolution : list, optional
        Point resolution to map between point resolution and mesh resolution, by default [4, 4, 40]
    mip_rescale : resolution difference between 
    """
    if cv is None:
        cv = cloudvolume.CloudVolume(
            client.info.segmentation_source(), use_https=True, bounded=False)

    # Get the closest adjacent point for the root id within the radius.
    pt = np.array(point) // mip_rescale
    offset = radius // (np.array(mip_rescale) * np.array(resolution))
    lx = np.array(pt) - offset
    ux = np.array(pt) + offset
    bbox = cloudvolume.Bbox(lx, ux)
    vol = cv.download(bbox,
                      segids=[root_id])
    vol = np.squeeze(vol)
    if not bool(np.any(vol > 0)):
        raise ValueError('No point of the root id is near the specified point')

    ctr = offset * point * resolution
    xyz = np.vstack(np.where(vol > 0)).T
    xyz_nm = xyz * mip_rescale * resolution

    ind = np.argmin(np.linalg.norm(xyz_nm-ctr, axis=1))
    closest_pt = vol.bounds.minpt + xyz[ind]

    # Look up the level 2 supervoxel for that id.
    closest_sv = int(cv.download_point(closest_pt, size=1))
    lvl2_id = client.chunkedgraph.get_root_id(closest_sv, level2=True)

    if return_point:
        return lvl2_id, closest_pt * mip_rescale * resolution
    else:
        return lvl2_id


def update_lvl2_skeleton(l2_sk, l2br_locs):
    verts = l2_sk.vertices
    verts[l2_sk.branch_points_undirected] = l2br_locs
    return skeleton.Skeleton(vertices=verts, edges=l2_sk.edges, remove_zero_length_edges=False)


def refine_skeleton(l2_sk, l2dict_reversed, cv, convert_to_nm=True, root_location=None):
    verts = l2_sk.vertices

    l2br_locs, missing_ids = lvl2_branch_fragment_locs(
        l2_sk, l2dict_reversed, cv)
    # if np.any(np.isnan(l2br_locs)):
    #     bad_l2_inds = np.any(np.isnan(l2br_locs), axis=1)
    #     bad_minds = l2_sk.mesh_index[l2_sk.branch_points_undirected[bad_l2_inds]]
    #     bad_l2_ids = [l2dict_reversed[mind] for mind in bad_minds]
    #     raise ValueError(f'No mesh found for lvl2 ids: {bad_l2_ids}')

    missing_brinds = np.any(np.isnan(l2br_locs), axis=1)
    verts[l2_sk.branch_points_undirected[~missing_brinds]
          ] = l2br_locs[~missing_brinds]

    if convert_to_nm:
        other_inds = np.full(len(verts), True)
        other_inds[l2_sk.branch_points_undirected[~missing_brinds]] = False
        verts[other_inds] = chunk_to_nm(
            verts[other_inds], cv) + chunk_dims(cv) // 2   # Move to center of chunks

    if root_location is not None:
        verts[l2_sk.root] = root_location

    return skeleton.Skeleton(vertices=verts, edges=l2_sk.edges, root=l2_sk.root, remove_zero_length_edges=False), missing_ids


# def generate_lvl2_proofreading(datastack, root_id, invalidation_d=3):
#     l2br_locs, l2_sk = get_lvl2_branch_points(
#         datastack, root_id, invalidation_d=invalidation_d, return_skeleton=True, verbose=True)
#     sk_pf = update_lvl2_skeleton(l2_sk, l2br_locs)
#     bp_statebuilder = lvl2_statebuilder(datastack, root_id)
#     drop_rows = np.any(np.isnan(l2br_locs), axis=1)

#     return bp_statebuilder.render_state(l2br_locs[~drop_rows] / np.array([4, 4, 40]), return_as='url')


def root_sb_data(sk, set_position=False, active=False, layer_name='root', voxel_resolution=[4, 4, 40]):
    pt = sb.PointMapper(point_column='pt', set_position=set_position)
    root_layer = sb.AnnotationLayerConfig(
        layer_name, color='#bfae00', mapping_rules=[pt], active=active)
    root_sb = sb.StateBuilder([root_layer])
    root_df = pd.DataFrame({'pt': (np.atleast_2d(sk.vertices[sk.root]) / np.array(voxel_resolution)).tolist(),
                            })
    return root_sb, root_df


def generate_lvl2_proofreading(datastack, root_id, root_point=None, point_radius=200, invalidation_d=3, auto_remesh=True, return_as='url', verbose=True):
    if verbose:
        t0 = time.time()
    client = FrameworkClient(datastack)

    l2_sk, l2dict_reversed = get_lvl2_skeleton(client, root_id, root_point=root_point, refine_branch_points=True, convert_to_nm=True,
                                               point_radius=point_radius, invalidation_d=invalidation_d, verbose=verbose, auto_remesh=auto_remesh)

    sbs = []
    dfs = []

    base_sb, base_df = base_sb_data(
        client, root_id, focus_loc=root_point, fixed_ids=[root_id])
    sbs.append(base_sb)
    dfs.append(base_df)

    rt_sb, rt_df = root_sb_data(l2_sk, set_position=True)
    sbs.append(rt_sb)
    dfs.append(rt_df)

    lbls = process_node_groups(l2_sk, cp_max_thresh=200_000)
    bp_sb, bp_df = branch_sb_data(
        client, root_id, l2_sk, labels=lbls, tags=BP_PROOFREADING_TAGS, set_position=False, active=True)
    sbs.append(bp_sb)
    dfs.append(bp_df)
    sb_pf = sb.ChainedStateBuilder(sbs)
    if verbose:
        print('\nComplete time: ', time.time()-t0)
    return sb_pf.render_state(dfs, return_as=return_as, url_prefix=client.info.viewer_site())
