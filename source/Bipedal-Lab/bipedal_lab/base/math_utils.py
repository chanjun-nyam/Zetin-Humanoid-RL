from typing import Tuple

import torch as th



def quat_apply(quat: th.Tensor, vec: th.Tensor) -> th.Tensor:
    """Apply quaternion to vector.

    Args:
        quat (th.Tensor): Quaternion tensor which shape is (..., 4) and order is (w, x, y, z).
        vec (th.Tensor): Vector tensor which shape is (..., 3) and order is (x, y, z).

    Returns:
        th.Tensor: vector rotated by given quaternion
    """
    quat_v = quat[...,1:]
    quat_w = quat[...,:1]
    
    uvec = th.cross(quat_v, vec, dim=-1)
    uuvec = th.cross(quat_v, uvec, dim=-1)

    return vec + 2.0 * (th.mul(quat_w, uvec) + uuvec)



def quat_apply_inv(quat: th.Tensor, vec: th.Tensor) -> th.Tensor:
    """Apply inverse quaternion to vector.

    Args:
        quat (th.Tensor): Quaternion tensor which shape is (..., 4) and order is (w, x, y, z).
        vec (th.Tensor): Vector tensor which shape is (..., 3) and order is (x, y, z).

    Returns:
        th.Tensor: vector rotated by given inverse quaternion
    """
    quat_v = -quat[...,1:]
    quat_w = quat[...,:1]
    
    uvec = th.cross(quat_v, vec, dim=-1)
    uuvec = th.cross(quat_v, uvec, dim=-1)

    return vec + 2.0 * (th.mul(quat_w, uvec) + uuvec)



def twist_swing_decomposition(quat: th.Tensor, eps: float = 1e-6) -> Tuple[th.Tensor, th.Tensor]:
    r"""Decompose quaternion to twist component and swing component.

    Equation can be written as

    $$ q = q_\text{t} \cdot q_\text{s} $$

    where rotation axis of twist quaternion $q_\text{t}$ have only z component and only xy components for swing quaternion $q_\text{s}$.

    Args:
        quat (th.Tensor):
            Quaternion tensor with shape (..., 4).
            Order of elements in last dimention is (w, x, y, z).
        eps (float, optional): for numerical stability
    
    Returns:
        tuple of twist quaternion and swing quaternion
    """
    w = quat[...,0]
    x = quat[...,1]
    y = quat[...,2]
    z = quat[...,3]

    w_xy = (w.square() + z.square()).sqrt()

    w_xy_clip = th.maximum(w_xy, eps)
    w_xy = th.where(w_xy > eps, w_xy, 0.0)

    w_z = th.where(w_xy > eps, w / w_xy_clip, 1.0)
    z_z = th.where(w_xy > eps, z / w_xy_clip, 0.0)
    x_xy = th.mul(x, w_z) + th.mul(y, z_z)
    y_xy = th.mul(y, w_z) - th.mul(x, z_z)

    # twist quaternion
    quat_t = th.zeros_like(quat)
    quat_t[...,0] = w_z
    quat_t[...,3] = z_z

    # swing quaternion
    quat_s = th.zeros_like(quat)
    quat_s[...,0] = w_xy
    quat_s[...,1] = x_xy
    quat_s[...,2] = y_xy

    return quat_t, quat_s
