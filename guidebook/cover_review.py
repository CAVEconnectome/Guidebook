import pandas as pd
import numpy as np
from scipy import interpolate


def sample_end_points(sk, n_choice, ep_segment_thresh=0):
    eps = viable_end_points(sk, ep_segment_thresh=ep_segment_thresh)
    if n_choice == "all":
        return eps
    else:
        return np.random.choice(eps, n_choice, replace=False)


def viable_end_points(sk, ep_segment_thresh=0):
    """Determines which end points are good for sampling by allowing ones on short segments to be"""
    eps = []
    cp_len = []
    for cp in sk.cover_paths:
        eps.append(cp[0])
        cp_len.append(sk.path_length(cp) / 1000)
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
    paths, sk, interp_spacing=2500, interp_method="linear", voxel_resolution=[4, 4, 40]
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