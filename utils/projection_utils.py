# pyre-unsafe
from typing import Tuple

import torch

_TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])


def maybe_jit_script(fn):
    """Apply @torch.jit.script only on PyTorch >= 1.13."""
    if _TORCH_VERSION >= (1, 13):
        return torch.jit.script(fn)
    return fn


def sign_plus(x):
    """
    return +1 for positive and for 0.0 in x. This is important for our handling
    of z values that should never be 0.0
    """
    # Use pure math ops for TorchScript CPU compatibility
    # (torch.ones_like and in-place indexing cause "Global alloc not supported yet" on CPU)
    # For x >= 0: 2*1 - 1 = 1, for x < 0: 2*0 - 1 = -1
    return 2.0 * (x >= 0.0).to(x.dtype) - 1.0


def _fisheye624_project_impl(xyz, params, suppress_warning=False):
    """
    Batched implementation of the FisheyeRadTanThinPrism (aka Fisheye624) camera
    model project() function. Note: this implementation is based heavily off of the c++
    file "arvr/libraries/perception/camera/projection/FisheyeRadTanThinPrism.h".

    Inputs:
        xyz: Bx(T)xNx3 tensor of 3D points to be projected
        params: Bx(T)x16 tensor of Fisheye624 parameters formatted like this:
                [f_u f_v c_u c_v {k_0 ... k_5} {p_0 p_1} {s_0 s_1 s_2 s_3}]
                or Bx(T)x15 tensor of Fisheye624 parameters formatted like this:
                [f c_u c_v {k_0 ... k_5} {p_0 p_1} {s_0 s_1 s_2 s_3}]
    Outputs:
        uv: Bx(T)xNx2 tensor of 2D projections of xyz in image plane

    Model for fisheye cameras with radial, tangential, and thin-prism distortion.
    This model allows fu != fv.
    Specifically, the model is:
    uvDistorted = [x_r]  + tangentialDistortion  + thinPrismDistortion
                  [y_r]
    proj = diag(fu,fv) * uvDistorted + [cu;cv];
    where:
      a = x/z, b = y/z, r = (a^2+b^2)^(1/2)
      th = atan(r)
      cosPhi = a/r, sinPhi = b/r
      [x_r]  = (th+ k0 * th^3 + k1* th^5 + ...) [cosPhi]
      [y_r]                                     [sinPhi]
      the number of terms in the series is determined by the template parameter numK.
      tangentialDistortion = [(2 x_r^2 + rd^2)*p_0 + 2*x_r*y_r*p_1]
                             [(2 y_r^2 + rd^2)*p_1 + 2*x_r*y_r*p_0]
      where rd^2 = x_r^2 + y_r^2
      thinPrismDistortion = [s0 * rd^2 + s1 rd^4]
                            [s2 * rd^2 + s3 rd^4]

    Author: Daniel DeTone (ddetone)
    """

    assert (xyz.ndim == 3 and params.ndim == 2) or (
        xyz.ndim == 4 and params.ndim == 3
    ), f"point dim {xyz.shape} does not match cam parameter dim {params}"
    assert xyz.shape[-1] == 3
    assert params.shape[-1] == 16 or params.shape[-1] == 15, (
        "This model allows fx != fy"
    )

    # Warn if input magnitudes are large enough to cause float32 precision issues
    if xyz.dtype == torch.float32 and xyz.numel() > 0 and not suppress_warning:
        max_abs = xyz.abs().max().item()
        if max_abs > 1000:
            import traceback
            import warnings

            stack = "".join(traceback.format_stack())
            warnings.warn(
                f"fisheye624_project: large input magnitude ({max_abs:.0f}) may cause "
                f"float32 precision loss. Consider centering points before projection.\n"
                f"Called from:\n{stack}",
                stacklevel=3,
            )

    eps = 1e-9
    T = -1
    if xyz.ndim == 4:
        # has T dim
        T, N = xyz.shape[1], xyz.shape[2]
        xyz = xyz.reshape(-1, N, 3)  # (BxT)xNx3
        params = params.reshape(-1, params.shape[-1])  #  (BxT)x16

    B, N = xyz.shape[0], xyz.shape[1]

    # Radial correction.
    z = xyz[:, :, 2].reshape(B, N, 1)
    # Do not use torch.sign(z) it leads to 0.0 zs if z == 0.0 which leads to a
    # nan when we compute xy/z
    z = torch.where(torch.abs(z) < eps, eps * sign_plus(z), z)
    ab = xyz[:, :, :2] / z
    # make sure abs are not too small or 0 otherwise gradients are nan
    ab = torch.where(torch.abs(ab) < eps, eps * sign_plus(ab), ab)
    r = torch.norm(ab, dim=-1, p=2, keepdim=True)
    th = torch.atan(r)
    # Avoid torch.ones_like for TorchScript CPU compatibility
    ones_ab = torch.zeros_like(ab) + 1.0
    th_divr = torch.where(r < eps, ones_ab, ab / r)
    th_k = th.reshape(B, N, 1).clone()
    for i in range(6):
        th_k = th_k + params[:, -12 + i].reshape(B, 1, 1) * torch.pow(th, 3 + i * 2)
    xr_yr = th_k * th_divr
    uv_dist = xr_yr

    # Tangential correction.
    p0 = params[:, -6].reshape(B, 1)
    p1 = params[:, -5].reshape(B, 1)
    xr = xr_yr[:, :, 0].reshape(B, N)
    yr = xr_yr[:, :, 1].reshape(B, N)
    xr_yr_sq = torch.square(xr_yr)
    xr_sq = xr_yr_sq[:, :, 0].reshape(B, N)
    yr_sq = xr_yr_sq[:, :, 1].reshape(B, N)
    rd_sq = xr_sq + yr_sq
    uv_dist_tu = uv_dist[:, :, 0] + ((2.0 * xr_sq + rd_sq) * p0 + 2.0 * xr * yr * p1)
    uv_dist_tv = uv_dist[:, :, 1] + ((2.0 * yr_sq + rd_sq) * p1 + 2.0 * xr * yr * p0)
    uv_dist = torch.stack(
        [uv_dist_tu, uv_dist_tv], dim=-1
    )  # Avoids in-place complaint.

    # Thin Prism correction.
    s0 = params[:, -4].reshape(B, 1)
    s1 = params[:, -3].reshape(B, 1)
    s2 = params[:, -2].reshape(B, 1)
    s3 = params[:, -1].reshape(B, 1)
    rd_4 = torch.square(rd_sq)
    # Avoid in-place slice assignment for TorchScript CPU compatibility
    uv_dist_tp0 = uv_dist[:, :, 0] + (s0 * rd_sq + s1 * rd_4)
    uv_dist_tp1 = uv_dist[:, :, 1] + (s2 * rd_sq + s3 * rd_4)
    uv_dist = torch.stack([uv_dist_tp0, uv_dist_tp1], dim=-1)

    # Finally, apply standard terms: focal length and camera centers.
    if params.shape[-1] == 15:
        fx_fy = params[:, 0].reshape(B, 1, 1)
        cx_cy = params[:, 1:3].reshape(B, 1, 2)
    else:
        fx_fy = params[:, 0:2].reshape(B, 1, 2)
        cx_cy = params[:, 2:4].reshape(B, 1, 2)
    result = uv_dist * fx_fy + cx_cy

    if T > 0:
        result = result.reshape(B // T, T, N, 2)

    assert result.ndim == 4 or result.ndim == 3
    assert result.shape[-1] == 2

    return result


def fisheye624_project(xyz, params, suppress_warning=False):
    """
    Public interface for fisheye624 projection.

    Note: TorchScript (@torch.jit.script) has been removed due to
    "Global alloc not supported yet" errors on CPU. The performance
    impact is minimal since this is typically called during visualization
    which is not in the critical training path.
    """
    return _fisheye624_project_impl(xyz, params, suppress_warning=suppress_warning)


@maybe_jit_script
def fisheye624_unproject(uv, params, max_iters: int = 5):
    """
    Batched implementation of the FisheyeRadTanThinPrism (aka Fisheye624) camera
    model. There is no analytical solution for the inverse of the project()
    function so this solves an optimization problem using Newton's method to get
    the inverse. Note: this implementation is based heavily off of the c++
    file "arvr/libraries/perception/camera/projection/FisheyeRadTanThinPrism.h".

    Inputs:
        uv: Bx(T)xNx2 tensor of 2D pixels to be projected
        params: Bx(T)x16 tensor of Fisheye624 parameters formatted like this:
                [f_u f_v c_u c_v {k_0 ... k_5} {p_0 p_1} {s_0 s_1 s_2 s_3}]
                or Bx(T)x15 tensor of Fisheye624 parameters formatted like this:
                [f c_u c_v {k_0 ... k_5} {p_0 p_1} {s_0 s_1 s_2 s_3}]
    Outputs:
        xyz: Bx(T)xNx3 tensor of 3D rays of uv points with z = 1.

    Model for fisheye cameras with radial, tangential, and thin-prism distortion.
    This model assumes fu=fv. This unproject function holds that:

    X = unproject(project(X))     [for X=(x,y,z) in R^3, z>0]

    and

    x = project(unproject(s*x))   [for s!=0 and x=(u,v) in R^2]

    Author: Daniel DeTone (ddetone)
    """
    # Note(nyn): The unprojection sometimes results in NaNs when using Float32.
    #            A temporary workaround in Perveiver is passing in Float64 (double) parameters, see: fbcode/surreal/perceiver/utils/encoding_utils.py.

    assert uv.ndim == 3 or uv.ndim == 4, "Expected batched input shaped Bx(T)xNx2"
    assert uv.shape[-1] == 2
    assert params.ndim == 2 or params.ndim == 3, (
        "Expected batched input shaped Bx(T)x16 or Bx(T)x15"
    )
    assert params.shape[-1] == 16 or params.shape[-1] == 15, (
        "This model allows fx != fy"
    )
    eps = 1e-6

    T = -1
    if uv.ndim == 4:
        # has T dim
        T, N = uv.shape[1], uv.shape[2]
        uv = uv.reshape(-1, N, 2)  # (BxT)xNx2
        params = params.reshape(-1, params.shape[-1])  #  (BxT)x16

    B, N = uv.shape[0], uv.shape[1]

    if params.shape[-1] == 15:
        fx_fy = params[:, 0].reshape(B, 1, 1)
        cx_cy = params[:, 1:3].reshape(B, 1, 2)
    else:
        fx_fy = params[:, 0:2].reshape(B, 1, 2)
        cx_cy = params[:, 2:4].reshape(B, 1, 2)

    uv_dist = (uv - cx_cy) / fx_fy

    # Compute xr_yr using Newton's method.
    xr_yr = uv_dist.clone()  # Initial guess.
    for _ in range(max_iters):
        uv_dist_est = xr_yr.clone()
        # Tangential terms.
        p0 = params[:, -6].reshape(B, 1)
        p1 = params[:, -5].reshape(B, 1)
        xr = xr_yr[:, :, 0].reshape(B, N)
        yr = xr_yr[:, :, 1].reshape(B, N)
        xr_yr_sq = torch.square(xr_yr)
        xr_sq = xr_yr_sq[:, :, 0].reshape(B, N)
        yr_sq = xr_yr_sq[:, :, 1].reshape(B, N)
        rd_sq = xr_sq + yr_sq
        uv_dist_est[:, :, 0] = uv_dist_est[:, :, 0] + (
            (2.0 * xr_sq + rd_sq) * p0 + 2.0 * xr * yr * p1
        )
        uv_dist_est[:, :, 1] = uv_dist_est[:, :, 1] + (
            (2.0 * yr_sq + rd_sq) * p1 + 2.0 * xr * yr * p0
        )
        # Thin Prism terms.
        s0 = params[:, -4].reshape(B, 1)
        s1 = params[:, -3].reshape(B, 1)
        s2 = params[:, -2].reshape(B, 1)
        s3 = params[:, -1].reshape(B, 1)
        rd_4 = torch.square(rd_sq)
        uv_dist_est[:, :, 0] = uv_dist_est[:, :, 0] + (s0 * rd_sq + s1 * rd_4)
        uv_dist_est[:, :, 1] = uv_dist_est[:, :, 1] + (s2 * rd_sq + s3 * rd_4)
        # Compute the derivative of uv_dist w.r.t. xr_yr.
        duv_dist_dxr_yr = uv.new_ones(B, N, 2, 2)
        duv_dist_dxr_yr[:, :, 0, 0] = (
            1.0 + 6.0 * xr_yr[:, :, 0] * p0 + 2.0 * xr_yr[:, :, 1] * p1
        )
        offdiag = 2.0 * (xr_yr[:, :, 0] * p1 + xr_yr[:, :, 1] * p0)
        duv_dist_dxr_yr[:, :, 0, 1] = offdiag
        duv_dist_dxr_yr[:, :, 1, 0] = offdiag
        duv_dist_dxr_yr[:, :, 1, 1] = (
            1.0 + 6.0 * xr_yr[:, :, 1] * p1 + 2.0 * xr_yr[:, :, 0] * p0
        )
        xr_yr_sq_norm = xr_yr_sq[:, :, 0] + xr_yr_sq[:, :, 1]
        temp1 = 2.0 * (s0 + 2.0 * s1 * xr_yr_sq_norm)
        duv_dist_dxr_yr[:, :, 0, 0] = duv_dist_dxr_yr[:, :, 0, 0] + (
            xr_yr[:, :, 0] * temp1
        )
        duv_dist_dxr_yr[:, :, 0, 1] = duv_dist_dxr_yr[:, :, 0, 1] + (
            xr_yr[:, :, 1] * temp1
        )
        temp2 = 2.0 * (s2 + 2.0 * s3 * xr_yr_sq_norm)
        duv_dist_dxr_yr[:, :, 1, 0] = duv_dist_dxr_yr[:, :, 1, 0] + (
            xr_yr[:, :, 0] * temp2
        )
        duv_dist_dxr_yr[:, :, 1, 1] = duv_dist_dxr_yr[:, :, 1, 1] + (
            xr_yr[:, :, 1] * temp2
        )
        # Compute 2x2 inverse manually here since torch.inverse() is very slow.
        # Because this is slow: inv = duv_dist_dxr_yr.inverse()
        # About a 10x reduction in speed with above line.
        mat = duv_dist_dxr_yr.reshape(-1, 2, 2)
        a = mat[:, 0, 0].reshape(-1, 1, 1)
        b = mat[:, 0, 1].reshape(-1, 1, 1)
        c = mat[:, 1, 0].reshape(-1, 1, 1)
        d = mat[:, 1, 1].reshape(-1, 1, 1)
        det = 1.0 / ((a * d) - (b * c))
        top = torch.cat([d, -b], dim=2)
        bot = torch.cat([-c, a], dim=2)
        inv = det * torch.cat([top, bot], dim=1)
        inv = inv.reshape(B, N, 2, 2)
        # Manually compute 2x2 @ 2x1 matrix multiply.
        # Because this is slow: step = (inv @ (uv_dist - uv_dist_est)[..., None])[..., 0]
        diff = uv_dist - uv_dist_est
        a = inv[:, :, 0, 0]
        b = inv[:, :, 0, 1]
        c = inv[:, :, 1, 0]
        d = inv[:, :, 1, 1]
        e = diff[:, :, 0]
        f = diff[:, :, 1]
        step = torch.stack([a * e + b * f, c * e + d * f], dim=-1)
        # Newton step.
        xr_yr = xr_yr + step

    # Compute theta using Newton's method.
    xr_yr_norm = xr_yr.norm(p=2, dim=2).reshape(B, N, 1)
    th = xr_yr_norm.clone()
    for _ in range(max_iters):
        # nyn: fix according to https://fburl.com/code/a87oq7yb.
        th_radial = uv.new_ones(B, N, 1)
        dthd_th = uv.new_ones(B, N, 1)
        for k in range(6):
            r_k = params[:, -12 + k].reshape(B, 1, 1)
            th_radial = th_radial + (r_k * torch.pow(th, 2 + k * 2))
            dthd_th = dthd_th + ((3.0 + 2.0 * k) * r_k * torch.pow(th, 2 + k * 2))
        th_radial = th_radial * th
        step = (xr_yr_norm - th_radial) / dthd_th
        # handle dthd_th close to 0.
        step = torch.where(dthd_th.abs() > eps, step, sign_plus(step) * eps * 10.0)
        th = th + step
    # Compute the ray direction using theta and xr_yr.
    close_to_zero = torch.logical_and(th.abs() < eps, xr_yr_norm.abs() < eps)
    ray_dir = torch.where(close_to_zero, xr_yr, torch.tan(th) / xr_yr_norm * xr_yr)
    ray = torch.cat([ray_dir, uv.new_ones(B, N, 1)], dim=2)
    assert ray.shape[-1] == 3

    if T > 0:
        ray = ray.reshape(B // T, T, N, 3)

    return ray


def pinhole_project(xyz, params):
    """
    Batched implementation of the Pinhole (aka Linear) camera
    model project() function. Note: this implementation is based heavily off of the c++
    file "arvr/libraries/perception/camera/projection/Pinhole.h".

    Inputs:
        xyz: Bx(T)xNx3 tensor of 3D points to be projected
        params: Bx(T)x4 tensor of Pinhole parameters formatted like this:
                [f_u f_v c_u c_v]
    Outputs:
        uv: Bx(T)xNx2 tensor of 2D projections of xyz in image plane
    """

    assert (xyz.ndim == 3 and params.ndim == 2) or (xyz.ndim == 4 and params.ndim == 3)
    assert params.shape[-1] == 4
    eps = 1e-9

    # Focal length and principal point
    fx_fy = params[..., 0:2].reshape(*xyz.shape[:-2], 1, 2)
    cx_cy = params[..., 2:4].reshape(*xyz.shape[:-2], 1, 2)
    # Make sure depth is not too close to zero.
    z = xyz[..., 2:]
    # Do not use torch.sign(z) it leads to 0.0 zs if z == 0.0 which leads to a
    # nan when we compute xy/z
    z = torch.where(torch.abs(z) < eps, eps * sign_plus(z), z)
    uv = (xyz[..., :2] / z) * fx_fy + cx_cy
    return uv


def pinhole_unproject(uv, params, max_iters: int = 5):
    """
    Batched implementation of the Pinhole (aka Linear) camera
    model. This implementation is based heavily off of the c++
    file "arvr/libraries/perception/camera/projection/Pinhole.h".

    Inputs:
        uv: Bx(T)xNx2 tensor of 2D pixels to be projected
        params: Bx(T)x4 tensor of Pinhole parameters formatted like this:
                [f_u f_v c_u c_v]
    Outputs:
        xyz: Bx(T)xNx3 tensor of 3D rays of uv points with z = 1.

    """
    assert uv.ndim == 3 or uv.ndim == 4, "Expected batched input shaped Bx(T)xNx3"
    assert params.ndim == 2 or params.ndim == 3
    assert params.shape[-1] == 4
    assert uv.shape[-1] == 2

    # Focal length and principal point
    fx_fy = params[..., 0:2].reshape(*uv.shape[:-2], 1, 2)
    cx_cy = params[..., 2:4].reshape(*uv.shape[:-2], 1, 2)

    uv_dist = (uv - cx_cy) / fx_fy

    ray = torch.cat([uv_dist, uv.new_ones(*uv.shape[:-1], 1)], dim=-1)
    return ray


def brown_conrady_project(xyz, params):
    """
    Batched implementation of the Brown Conrady radial camera

    Inputs:
        xyz: Bx(T)xNx3 tensor of 3D points to be projected
        # params: Bx(T)x4 tensor of Pinhole parameters formatted like this:
        #         [f_u f_v c_u c_v]

        params: Bx(T)x12 tensor of Fisheye624 parameters formatted like this:
                [f_u f_v c_u c_v {k_0 k_1 k_2 k_3} {p_0 p_1 p_2 p_3}]
                or Bx(T)x11 tensor of Fisheye624 parameters formatted like this:
                [f c_u c_v {k_0 k_1 k_2 k_3}{p_0 p_1 p_2 p_3}]
    Outputs:
        uv: Bx(T)xNx2 tensor of 2D projections of xyz in image plane
    """

    assert (xyz.ndim == 3 and params.ndim == 2) or (xyz.ndim == 4 and params.ndim == 3)
    assert params.shape[-1] == 4
    eps = 1e-9

    # Focal length and principal point
    fx_fy = params[..., 0:2].reshape(*xyz.shape[:-2], 1, 2)
    cx_cy = params[..., 2:4].reshape(*xyz.shape[:-2], 1, 2)

    B, N = xyz.shape[0], xyz.shape[1]

    # Make sure depth is not too close to zero.
    z = xyz[..., 2:]
    # Do not use torch.sign(z) it leads to 0.0 zs if z == 0.0 which leads to a
    # nan when we compute xy/z
    z = torch.where(torch.abs(z) < eps, eps * sign_plus(z), z)

    ab = xyz[:, :, :2] / z
    r = torch.norm(ab, dim=-1, p=2, keepdim=True)

    # radial correction terms
    r_k = torch.ones_like(r)
    for i in range(4):
        pow = 2 + i * 2
        r_k = r_k + params[:, 4 + i].reshape(B, 1, 1) * torch.pow(r, pow)

    print("===> warning untested work in progress!!!")
    uv_dist = xyz[..., :2] / z
    uv = (uv_dist * r_k) * fx_fy + cx_cy
    return uv


def kb4_project(
    rays: torch.Tensor,
    params: torch.Tensor,
    z_clip: float = 1e-8,
) -> torch.Tensor:
    """
    Batched KB4 projection (ray -> pixel).

    Args:
        rays:   (Bx(T)xNx3) camera-frame rays
        params: (Bx(T)x8) [fu,fv,u0,v0,k0,k1,k2,k3]
        z_clip: clamp Z to avoid divide-by-zero

    Returns:
        uv: (B,N,2) pixel coordinates
    """
    print(f"==> warning kb4 project, work in progress!!!")
    assert rays.ndim == 3 or rays.ndim == 4
    assert rays.size(-1) == 3
    assert params.ndim == 2 or params.ndim == 3
    assert params.size(-1) == 8
    # unpack
    fu, fv, u0, v0, k0, k1, k2, k3 = params.unbind(-1)
    X, Y, Z = rays.unbind(-1)
    Z = Z.clamp_min(z_clip)
    # normalized coordinates
    xn, yn = X / Z, Y / Z
    rho = torch.sqrt(xn * xn + yn * yn).clamp_min(1e-12)
    theta = torch.atan(rho)
    # KB4 mapping: r(theta) = k0*θ + k1*θ^3 + k2*θ^5 + k3*θ^7
    t2 = theta * theta
    r = (
        k0[..., None] * theta
        + k1[..., None] * theta * t2
        + k2[..., None] * theta * t2 * t2
        + k3[..., None] * theta * t2 * t2 * t2
    )
    # scale back to pixel plane
    scale = r / rho
    xd, yd = xn * scale, yn * scale
    u = u0[..., None] + fu[..., None] * xd
    v = v0[..., None] + fv[..., None] * yd
    return torch.stack([u, v], dim=-1)  # (B,N,2)


def kb4_unproject(
    uv: torch.Tensor,
    params: torch.Tensor,
    iters: int = 5,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Batched unprojection for Kannala–Brandt KB4 camera model.
    Args:
        uv:     (Bx(T)xNx2) pixel coordinates
        params: (Bx(T)x8)   [fu,fv,u0,v0,k0,k1,k2,k3]
        iters:  Newton iterations for solving theta
        z_clip: clamp for numerical stability
        tol:    convergence threshold for valid mask
    Returns:
        rays:  (B,N,3) unit-length rays in camera coordinates
    """
    print(f"==> warning kb4 unproject, work in progress!!!")
    assert uv.ndim == 3 or uv.ndim == 4
    assert uv.size(-1) == 2
    assert params.ndim == 2 or params.ndim == 3
    assert params.size(-1) == 8

    fu, fv, u0, v0, k0, k1, k2, k3 = params.unbind(-1)
    u, v = uv[..., 0], uv[..., 1]

    # normalized distorted image plane coords
    xd = (u - u0[..., None]) / fu[..., None]
    yd = (v - v0[..., None]) / fv[..., None]
    rd = torch.sqrt(xd * xd + yd * yd).clamp_min(eps)

    # initial guess for theta
    # theta = rd / k0[:, None].clamp_min(eps)
    theta = torch.atan(rd)

    for ii in range(iters):
        t2 = theta * theta
        # forward KB4 mapping
        r_theta = (
            k0[..., None] * theta
            + k1[..., None] * theta * t2
            + k2[..., None] * theta * t2 * t2
            + k3[..., None] * theta * t2 * t2 * t2
        )
        f = r_theta - rd
        # derivative wrt theta
        dr_dtheta = (
            k0[..., None]
            + 3 * k1[..., None] * t2
            + 5 * k2[..., None] * t2 * t2
            + 7 * k3[..., None] * t2 * t2 * t2
        )
        theta = theta - (f / dr_dtheta)
        print(f"==> iteration {ii}/{iters}")
        print(f"theta = {theta}")
        print(f"r_theta = {r_theta}")
        print(f"rd = {rd}")
        print(f"f = {f}")
        print(f"dr_dtheta = {dr_dtheta}")

    # final radius on normalized plane
    rho = torch.tan(theta)
    dirx = xd / (rd * rho + eps)
    diry = yd / (rd * rho + eps)
    rays = torch.stack([dirx, diry, torch.ones_like(dirx)], dim=-1)
    denom = torch.linalg.norm(rays, dim=-1, keepdim=True)
    rays = rays / denom

    return rays
