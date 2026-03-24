# pyre-unsafe
from bisect import bisect_left

import numpy as np
import torch


def sample_nearest(value_a, value_b, array_b):
    array_b_at_a = []
    for v_a in value_a:
        idx = find_nearest(value_b, v_a, return_index=True)
        array_b_at_a.append(array_b[idx])
    return torch.stack(array_b_at_a)


def find_nearest(array, value, return_index=False, warning_if_beyond=False):
    """Find the index of the nearest value in a numpy array."""
    array = np.asarray(array)
    val = np.abs(array - value)
    idx = (val).argmin()
    if return_index:
        return idx
    else:
        return array[idx]


def find_nearest2(array, value):
    """Find the index of the nearest value in an array."""
    idx = bisect_left(array, value)
    if idx == len(array):
        return idx - 1
    if idx == 0:
        return 0
    before = array[idx - 1]
    after = array[idx]
    if after - value < value - before:
        return idx
    return idx - 1


def find_upper_nearest(array, value, return_index=False):
    array = np.asarray(array)
    diff = array - value
    max_value = np.abs(diff).max() + 1000
    diff_upper = np.where(diff < 0.0, np.ones_like(diff) * max_value, diff)
    idx_upper = diff_upper.argmin()
    if return_index:
        return idx_upper
    else:
        return array[idx_upper]


def find_lower_nearest(array, value, return_index=False):
    array = np.asarray(array)
    diff = array - value
    max_value = np.abs(diff).max() + 1000
    diff_lower = np.where(diff > 0.0, np.ones_like(diff) * max_value, -diff)
    idx_lower = diff_lower.argmin()
    if return_index:
        return idx_lower
    else:
        return array[idx_lower]


def find_lower_upper_nearest(array, value, return_index=False, max_value=None):
    array = np.asarray(array)
    diff = array - value
    if not max_value:
        max_value = np.abs(diff).max() + 1000
    diff_upper = np.where(diff < 0.0, np.ones_like(diff) * max_value, diff)
    diff_lower = np.where(diff > 0.0, np.ones_like(diff) * max_value, -diff)
    idx_upper = diff_upper.argmin()
    idx_lower = diff_lower.argmin()
    if return_index:
        if array[idx_lower] > array[idx_upper]:
            # value was beyond array.max()
            return idx_lower, idx_lower
        return idx_lower, idx_upper
    else:
        if array[idx_lower] > array[idx_upper]:
            # value was beyond array.max()
            return array[idx_lower], array[idx_lower]
        return array[idx_lower], array[idx_upper]


def bb2extent(bb):
    if bb.ndim == 1:
        bb = bb.reshape(1, -1)
    x_min = bb[:, 0].min()
    x_max = bb[:, 0].max()
    y_min = bb[:, 1].min()
    y_max = bb[:, 1].max()
    z_min = bb[:, 2].min()
    z_max = bb[:, 2].max()
    out = np.stack([x_min, x_max, y_min, y_max, z_min, z_max], axis=0)
    return out


def extent2bb(extent):
    if extent.ndim == 1:
        extent = extent.reshape(1, -1)

    x_min, x_max = extent[:, 0], extent[:, 1]
    y_min, y_max = extent[:, 2], extent[:, 3]
    z_min, z_max = extent[:, 4], extent[:, 5]
    arr = (
        [
            x_min,
            y_min,
            z_min,
            x_max,
            y_min,
            z_min,
            x_max,
            y_max,
            z_min,
            x_min,
            y_max,
            z_min,
            x_min,
            y_min,
            z_max,
            x_max,
            y_min,
            z_max,
            x_max,
            y_max,
            z_max,
            x_min,
            y_max,
            z_max,
        ],
    )
    if torch.is_tensor(extent):
        bb3d = torch.stack(arr, dim=-1).reshape(-1, 8, 3)
    elif isinstance(extent, np.ndarray):
        bb3d = np.stack(arr, axis=-1).reshape(-1, 8, 3)
    else:
        raise TypeError("Unknown type")

    return bb3d.squeeze()


def pad_string(string, max_len=200, silent=False):
    """Pad a string with "spaces" at the end, up to 200 by default."""
    string2 = string[:max_len]
    if len(string) > max_len and not silent:
        print("Warning: string will be truncated to %s" % string2)
    if len(string) > 0 and string2[-1] == " " and not silent:
        print("Warning: string ends with a space, this may be lost when unpadding")
    return "{message: <{width}}".format(message=string2, width=max_len)


def unpad_string(string, max_len=200):
    """Remove extra space padding at the end of the string."""
    return string.rstrip()


def string2tensor(string):
    """convert a python string into torch tensor of chars"""
    return torch.tensor([ord(s) for s in string]).byte()


def tensor2string(tensor, unpad=False):
    """convert a torch tensor of chars to python string"""

    def safe_chr(val):
        """Convert integer to character, handling invalid values"""
        try:
            # Clamp to valid Unicode range
            val = int(val)
            if val < 0 or val > 0x10FFFF:
                return ""  # Skip invalid characters
            return chr(val)
        except (ValueError, OverflowError):
            return ""  # Skip on error

    if tensor.ndim == 1:
        out = "".join([safe_chr(s) for s in tensor])
        if unpad:
            out = unpad_string(out)
        return out
    elif tensor.ndim == 2:
        out = []
        for ex in tensor:
            ex2 = "".join([safe_chr(s) for s in ex])
            if unpad:
                ex2 = unpad_string(ex2)
            out.append(ex2)
        return out
    else:
        raise ValueError("Higher dims >2 not supported")


def pad_points(points_in, max_num_point=25000):
    """Pad point matrix with nan at the end, return fixed size matrix.
    the last row will be:
    nan nan nan ... numValidRow
    """
    assert max_num_point >= 3

    if points_in.ndim == 1:
        points_in = points_in.reshape(-1, 3)

    points_padded = torch.zeros(
        (max_num_point, points_in.shape[1]), device=points_in.device
    )
    numValidRow = min(points_in.shape[0], max_num_point - 1)
    # if points_in.shape[0] > numValidRow:
    #    print(
    #        "Warning: input points will be truncated from %d to %d"
    #        % (points_in.shape[0], numValidRow)
    #    )

    points_padded[0:numValidRow, :] = points_in[0:numValidRow, :]
    points_padded[numValidRow:, :] = float("nan")  # all nan from numValidRow
    points_padded[-1, -1] = numValidRow
    return points_padded


def unpad_points(points_in, return_num_valid=False):
    """Remove extra nan padding at the end of the points
    the last row will be:
    nan nan nan ... numValidRow
    """
    assert points_in.shape[0] >= 3
    # if input format matches padding pattern, unpad; otherwise the input is not padded, directly return it.
    if (
        points_in.dim() == 2
        and points_in.shape[0] > 0
        and points_in.shape[1] > 0
        and torch.isnan(points_in[-1, :-1]).all()
    ):
        numValidRow = int(points_in[-1, -1])
        assert numValidRow <= points_in.shape[0]
        if return_num_valid:
            return points_in[0:numValidRow, :], numValidRow
        else:
            return points_in[0:numValidRow, :]
    elif (  # Support pts_std
        points_in.dim() == 1 and points_in.shape[0] > 0 and torch.isnan(points_in).any()
    ):
        numValidRow = int(points_in[-1])
        assert numValidRow <= points_in.shape[0]
        if return_num_valid:
            return points_in[0:numValidRow, :], numValidRow
        else:
            return points_in[0:numValidRow, :]
    else:
        if return_num_valid:
            return points_in
        else:
            return points_in, points_in.shape[0]


def pad_points2(points_in, max_num_point=25000, warn=False):
    """Pad point matrix (Nx3 or 3) with -1 for all invalid rows."""
    assert max_num_point >= 3
    if points_in.ndim == 1:
        points_in = points_in.reshape(-1, 3)
    assert points_in.ndim == 2
    assert points_in.shape[-1] == 3
    device = points_in.device
    N = points_in.shape[0]
    out = points_in.clone()
    if N < max_num_point:
        out = torch.cat(
            [out, -1 * torch.ones((max_num_point - N, 3), device=device)], dim=0
        )
    if N > max_num_point:
        if warn:
            print(
                "Warning: input points will be truncated from %d to %d"
                % (N, max_num_point)
            )
        out = out[:max_num_point, :]
    return out


def get_invalid_points(points_in):
    invalid = torch.all(points_in == -1, dim=-1)
    # assert torch.all(torch.sort(invalid).values, invalid), "Invalid points must be contiguous"
    return invalid


def point_dropout(points_in, prob_all=0.5, prob_none=0.05, drop_min=0.5, drop_max=1.0):
    """
    randomly dropout points from a padded point vector

    uniformly sample number of points to dropout between 0,1 with some extra chance to dropout
    all or keep all

    """
    assert points_in.ndim == 3
    # assert torch.all(points_in[:, -1, 0].isnan()), "points must have at least one pad"
    B, N, _ = points_in.shape

    # num_valids = points_in[:, -1, -1].int()
    # Sample vals in [keep_min, keep_max]
    drop_max = drop_max
    drop_min = drop_min
    drops = (drop_max - drop_min) * torch.rand(B) + drop_min
    drop_none = torch.rand(B) < prob_none
    drop_all = torch.rand(B) < prob_all  # If tie, choose dropout all.
    drops[drop_none] = 0.0
    drops[drop_all] = 1.0

    # Dropout each batch inpdependently.
    for b in range(B):
        # nv = num_valids[b]
        invalid = get_invalid_points(points_in[b])
        nv = points_in[b].shape[0] - int(invalid.sum())
        valid_range = torch.arange(nv)
        # Shuffle the valid points.
        rand_perms = torch.randperm(nv)
        points_in[b, valid_range] = points_in[b, rand_perms]
        # Set the last "num_drop" points to NaN
        num_drop = int(drops[b] * nv)
        drop_start = nv - num_drop
        drop_end = nv
        points_in[b, drop_start:drop_end] = -1

        ## Update the valid count
        # points_in[b, -1, -1] -= num_drop

    return points_in


def basic_mean(x, dim, valid, keepdim=False, eps=1e-6, invalid_mean_val=-1):
    count = torch.sum(valid, dim=dim, keepdim=True)  # B 1 C/1 D H W
    invalid = (~valid).expand_as(x)
    x[invalid] = 0.0
    mean = torch.sum(x, dim=dim, keepdim=True) / (count + eps)
    not_enough = count.expand_as(mean) < 1
    mean[not_enough] = invalid_mean_val
    if not keepdim:
        return mean.squeeze(dim), count.squeeze(dim)
    else:
        return mean, count


def myunique(x, dim=-1):
    unique, inverse = torch.unique(x, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([dim]), perm.flip([dim])
    return unique, inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm)


def get_img_keys(d):
    """Return keys in a dict that are exactly 'img0'...'img9'."""
    valid_keys = {f"img{i}" for i in range(10)}
    return [k for k in d.keys() if k in valid_keys]
