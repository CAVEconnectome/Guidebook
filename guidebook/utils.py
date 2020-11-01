import numpy as np
from trimesh import creation
from .base import chunk_to_nm, chunk_dims
from functools import reduce


def _tmat(xyz):
    """4x4 transformation matrix for just translation by xyz"""
    T = np.eye(4)
    T[0:3, 3] = np.array(xyz).reshape(1, 3)
    return T


def chunk_box(xyz, chunk_size=[1, 1, 1]):
    """Create a trimesh box for a single chunk"""
    xyz_offset = xyz + np.array(chunk_size) / 2
    return creation.box(chunk_size, _tmat(xyz_offset))


def chunk_mesh(xyz_ch, cv):
    """Get a trimesh for a set of chunk positions

    Parameters
    ----------
    xyz_ch : np.array
        Nx3 array of chunk indices
    cv : cloudvolume.CloudVolume
        Home cloudvolume object for the chunks.
    """
    verts_ch_nm = chunk_to_nm(xyz_ch, cv)
    dim = chunk_dims(cv)

    boxes = [chunk_box(xyz, dim) for xyz in verts_ch_nm]
    boxes_all = reduce(lambda a, b: a+b, boxes)
    return boxes_all
