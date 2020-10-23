import pandas as pd
import numpy as np
from annotationframeworkclient import FrameworkClient
from scipy import sparse
from meshparty import trimesh_io, skeletonize, mesh_filters
from nglui import statebuilder as sb

MIN_CC_SIZE = 1000
SK_KWARGS = dict(invalidation_d=8000,
                 collapse_function='sphere',
                 soma_radius=8000,
                 compute_radius=False,
                 shape_function='single')

EP_PROOFREADING_TAGS = []
BP_PROOFREADING_TAGS = ['checked', 'error']


def process_mesh(oid, client, mm, min_component_size=1000):
    """Download mesh and postprocess for link edges and dust"""
    mesh = mm.mesh(seg_id=oid)
    mesh.add_link_edges(seg_id=oid, client=client.chunkedgraph)
    mf = mesh_filters.filter_components_by_size(mesh, min_size=1000)
    meshf = mesh.apply_mask(mf)
    return meshf


def process_skeleton(mesh, soma_loc=None, root_loc=None, close_mean=10000, sk_kwargs={}):
    sk = skeletonize.skeletonize_mesh(mesh,
                                      soma_pt=soma_loc,
                                      **sk_kwargs)
    # TODO reroot to root_loc
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


def proofer_statebuilder(oid, client, bp_proofreading_tags=[], ep_proofreading_tags=[]):

    state_server = client.state.state_service_endpoint
    img = sb.ImageLayerConfig(client.info.image_source(),
                              contrast_controls=True,
                              black=0.35, white=0.7)
    seg = sb.SegmentationLayerConfig(client.info.segmentation_source(),
                                     fixed_ids=[oid],
                                     view_kws={'alpha_selected': 0.2, 'alpha_3d': 0.6})

    points_bp = sb.PointMapper(
        point_column='bp_locs', group_column='bp_group', set_position=True)
    bp_layer = sb.AnnotationLayerConfig(
        'branch_points', mapping_rules=points_bp,
        active=True, color='#299bff', tags=bp_proofreading_tags)

    points_ep = sb.PointMapper(point_column='ep_locs')
    ep_layer = sb.AnnotationLayerConfig(
        'end_points', mapping_rules=points_ep,
        color='#ffffff', tags=ep_proofreading_tags)

    sb_ep = sb.StateBuilder(layers=[img, seg, ep_layer],
                            state_server=state_server)
    sb_bp = sb.StateBuilder(layers=[bp_layer], view_kws={'layout': '3d'})

    sb_all = sb.ChainedStateBuilder([sb_ep, sb_bp])

    return sb_all


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
        oid, client, bp_proofreading_tags, ep_proofreading_tags)
    return pf_sb, sb_dfs


def generate_proofreading_state(datastack,
                                root_id,
                                branch_points=True,
                                end_points=True,
                                soma_point=None,
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
    skf = process_skeleton(mesh, soma_loc=soma_point, sk_kwargs=SK_KWARGS)
    pf_sb, sb_dfs = process_points_from_skeleton(root_id, skf, client)

    state = pf_sb.render_state(
        sb_dfs, return_as=return_as, url_prefix=client.info.viewer_site())

    return state
