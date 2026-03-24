# pyre-unsafe
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch

# Some globals for opencv drawing functions.
BLU = (255, 0, 0)
GRN = (0, 255, 0)
RED = (0, 0, 255)
WHT = (255, 255, 255)
BLK = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_PT = (5, 15)
FONT_SZ = 0.5
FONT_TH = 1.0


def string2color(string):
    string = string.lower()
    if string == "white":
        return WHT
    elif string == "green":
        return GRN
    elif string == "red":
        return RED
    elif string == "black":
        return BLK
    elif string == "blue":
        return BLU
    else:
        raise ValueError("input color string %s not supported" % string)


def put_text(
    img: np.ndarray,
    text: str,
    scale: float = 1.0,
    line: int = 0,
    # pyre-fixme[9]: color has type `Tuple[tuple[Any, ...], str]`; used as
    #  `Tuple[int, int, int]`.
    color: Tuple[Tuple, str] = WHT,
    font_pt: Optional[Tuple[int, int]] = None,
    # pyre-fixme[9]: truncate has type `int`; used as `None`.
    truncate: int = None,
):
    """Writes text with a shadow in the back at various lines and autoscales it.

    Args:
        image: image HxWx3 or BxHxWx3, should be uint8 for anti-aliasing to work
        text: text to write
        scale: 0.5 for small, 1.0 for normal, 1.5 for big font
        line: vertical line to write on (0: first, 1: second, -1: last, etc)
        color: text color, tuple of BGR integers between 0-255, e.g. (0,0,255) is red,
               can also be a few strings like "white", "black", "green", etc
        truncate: if not None, only show the first N characters
    Returns:
        image with text drawn on it

    """
    if len(img.shape) == 4:  # B x H x W x 3
        for i in range(len(img)):
            img[i] = put_text(img[i], text, scale, line, color, font_pt, truncate)
    else:  # H x W x 3
        if truncate and len(text) > truncate:
            text = text[:truncate] + "..."  # Add "..." to denote truncation.
        height = img.shape[0]
        scale = scale * (height / 320.0)
        wht_th = max(int(FONT_TH * scale), 1)
        blk_th = 2 * wht_th
        text_ht = 15 * scale
        if not font_pt:
            font_pt = int(FONT_PT[0] * scale), int(FONT_PT[1] * scale)
            font_pt = font_pt[0], int(font_pt[1] + line * text_ht)
        if line < 0:
            font_pt = font_pt[0], int(font_pt[1] + (height - text_ht * 0.5))
        cv2.putText(img, text, font_pt, FONT, FONT_SZ * scale, BLK, blk_th, lineType=16)

        if isinstance(color, str):
            color = string2color(color)

        cv2.putText(
            img, text, font_pt, FONT, FONT_SZ * scale, color, wht_th, lineType=16
        )
    return img


def rotate_image90(image: np.ndarray, k: int = 3):
    """Rotates an image and then re-allocates memory to avoid problems with opencv
    Input:
        image: numpy image, HxW or HxWxC
        k: number of times to rotate by 90 degrees counter clockwise
    Returns
        rotated image: numpy image, HxW or HxWxC
    """
    return np.ascontiguousarray(np.rot90(image, k=k))


def normalize(img, robust=0.0, eps=1e-6):
    if isinstance(img, torch.Tensor):
        vals = img.view(-1).cpu().numpy()
    elif isinstance(img, np.ndarray):
        vals = img.flatten()

    if robust > 0.0:
        v_min = np.quantile(vals, robust)
        v_max = np.quantile(vals, 1.0 - robust)
    else:
        v_min = vals.min()
        v_max = vals.max()
    # make sure we are not dividing by 0
    dv = max(eps, v_max - v_min)
    # normalize to 0-1
    img = (img - v_min) / dv
    if isinstance(img, torch.Tensor):
        img = img.clamp(0, 1)
    elif isinstance(img, np.ndarray):
        img = img.clip(0, 1)
    return img


def torch2cv2(
    img: Union[np.ndarray, torch.Tensor],
    rotate: bool = False,
    rgb2bgr: bool = True,
    ensure_rgb: bool = False,
    robust_quant: float = 0.0,
):
    """
    Converts numpy/torch float32 image [0,1] CxHxW to numpy uint8 [0,255] HxWxC

    Args:
        img: image CxHxW float32 image
        rotate: if True, rotate image 90 degrees
        rgb2bgr: convert image to BGR
        ensure_rgb: ensure RGB if True (i.e. replicate the single color channel 3 times)
        robust_quant: quantile to robustly copute min and max for normalization of the image.
    """

    if isinstance(img, torch.Tensor):
        if img.dim() == 4:
            img = img[0]
        img = img.data.cpu().numpy()
    if img.ndim == 2:
        img = img[np.newaxis, :, :]

    # CxHxW -> HxWxC
    img = img.transpose(1, 2, 0)
    img_cv2 = (img * 255.0).astype(np.uint8)

    if rgb2bgr:
        img_cv2 = img_cv2[:, :, ::-1]
    if rotate:
        img_cv2 = rotate_image90(img_cv2)
    else:
        img_cv2 = np.ascontiguousarray(img_cv2)
    if ensure_rgb and img_cv2.shape[2] == 1:
        img_cv2 = img_cv2[:, :, 0]
    if ensure_rgb and img_cv2.ndim == 2:
        img_cv2 = np.stack([img_cv2, img_cv2, img_cv2], -1)
    return img_cv2
