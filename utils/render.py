# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

import cv2
import numpy as np
import torch

from .camera import CameraTW
from .image import put_text
from .obb import ObbTW
from .pose import PoseTW
from .tensor_utils import tensor2string, unpad_string

AXIS_COLORS_RGB = {
    0: (255, 0, 0),  # red
    3: (0, 255, 0),  # green
    8: (0, 0, 255),  # blue
}  # use RGB for xyz axes respectively


def draw_bb3_lines(
    viz,
    T_world_cam: PoseTW,
    cam: CameraTW,
    obbs: ObbTW,
    draw_cosy: bool,
    T: int,
    line_type=cv2.LINE_AA,
    thickness=1,
    prob_color=False,
    colors=None,
):
    bb3corners_world = obbs.T_world_object * obbs.bb3edge_pts_object(T)
    bb3corners_cam = T_world_cam.inverse() * bb3corners_world
    B = bb3corners_cam.shape[0]
    pt3s_cam = bb3corners_cam.view(B, -1, 3)
    pt2s, valids = cam.project(pt3s_cam)
    sem_ids = obbs.sem_id.int()
    # reshape to lines each composed of T segments
    pt2s = pt2s.round().int().view(B * 12, T, 2)
    valids = valids.view(B * 12, T)
    for line in range(pt2s.shape[0]):
        line_id = line % 12
        obb_id = line // 12
        sem_id = sem_ids[obb_id]
        if colors is not None:
            color = colors[obb_id]
        elif prob_color:
            prob = float(1.0 - obbs[obb_id].prob)
            max_val = 0.5
            min_val = 0.05
            prob = (prob - min_val) / (max_val - min_val)
            prob = max(0.0, min(1.0, prob))
            val = np.uint8([[int(prob * 255)]])
            bgr = cv2.applyColorMap(val, cv2.COLORMAP_JET)[0, 0]
            color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
        else:
            color = obbs[obb_id].color
            if (color == -1).all():
                color = 255, 255, 255
            else:
                color = (
                    int(round(float(color[2] * 255))),
                    int(round(float(color[1] * 255))),
                    int(round(float(color[0] * 255))),
                )

        for i in range(T - 1):
            j = i + 1
            if valids[line, i] and valids[line, j]:
                # check if we should color this line in a special way
                if draw_cosy and line_id in AXIS_COLORS_RGB:
                    color = AXIS_COLORS_RGB[line_id]
                pt1 = (
                    int(round(float(pt2s[line, i, 0]))),
                    int(round(float(pt2s[line, i, 1]))),
                )
                pt2 = (
                    int(round(float(pt2s[line, j, 0]))),
                    int(round(float(pt2s[line, j, 1]))),
                )
                cv2.line(
                    viz,
                    pt1,
                    pt2,
                    color,
                    thickness,
                    lineType=line_type,
                )


def draw_bb3s(
    viz,
    T_world_rig: PoseTW,
    cam: CameraTW,
    obbs: ObbTW,
    draw_bb3_center=False,
    draw_label=False,
    draw_cosy=False,
    draw_score=False,
    render_obb_corner_steps=10,
    line_type=cv2.LINE_AA,
    rotate_label=True,
    white_backing_line=False,
    already_rotated=False,
    prob_color=False,
    colors=None,
    texts=None,
    text_sz=0.35,
    thickness=1,
):
    if obbs.shape[0] == 0:
        return viz

    if already_rotated:
        viz = cv2.rotate(viz, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Get pose of camera.
    T_world_cam = T_world_rig.float() @ cam.T_camera_rig.inverse()

    # draw semantic colors
    draw_bb3_lines(
        viz,
        T_world_cam,
        cam,
        obbs,
        draw_cosy=draw_cosy,
        T=render_obb_corner_steps,
        line_type=cv2.LINE_AA,
        thickness=thickness,
        prob_color=prob_color,
        colors=colors,
    )

    if draw_label or draw_bb3_center or texts is not None:
        bb3center_cam = T_world_cam.inverse() * obbs.bb3_center_world
        bb2center_im, valids = cam.unsqueeze(0).project(bb3center_cam.unsqueeze(0))
        bb2center_im, valids = bb2center_im.squeeze(0), valids.squeeze(0)
        for idx, (pt2, valid) in enumerate(zip(bb2center_im, valids)):
            if valid:
                center = (int(pt2[0]), int(pt2[1]))
                if draw_bb3_center:
                    cv2.circle(viz, center, 3, (255, 0, 0), 1, lineType=line_type)

                if draw_label or texts is not None:
                    height = viz.shape[0]
                    sem_id = int(obbs.sem_id.squeeze(-1)[idx])

                    if texts is not None:
                        text = texts[idx]
                    else:
                        text = obbs.text[idx]
                        if (text == -1).all():
                            text = str(sem_id)
                        else:
                            text = unpad_string(tensor2string(obbs.text[idx].byte()))
                    if colors is not None:
                        text_clr = colors[idx]
                    else:
                        text_clr = (200, 200, 200)

                    # rot 90 degree before drawing the text
                    if rotate_label:
                        viz = cv2.rotate(viz, cv2.ROTATE_90_CLOCKWISE)
                        center_rot90 = (height - center[1], center[0])
                        x, y = center_rot90
                    else:
                        x, y = center
                    ((txt_w, txt_h), _) = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_DUPLEX, text_sz, 1
                    )

                    ## Show text on top of the 3d boxes
                    put_text(viz, text, scale=text_sz, font_pt=(x, y), color=text_clr)
                    if draw_score and obbs.prob is not None:
                        score = float(obbs.prob.squeeze(-1)[idx])
                        score_text = f"prob={score:.2f}"
                        score_pos = (x, y + int(txt_h + 0.5))
                        put_text(
                            viz,
                            score_text,
                            scale=text_sz,
                            font_pt=score_pos,
                            color=text_clr,
                        )

                    if rotate_label:
                        viz = cv2.rotate(viz, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if already_rotated:
        viz = cv2.rotate(viz, cv2.ROTATE_90_CLOCKWISE)

    return viz


def render_bb2(img, bb2s, scale=1.0, clr=(0, 255, 0), rotated=False, texts=None):
    if rotated:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if texts is not None:
        assert len(texts) == len(bb2s)

    if isinstance(clr, tuple):
        colors = [clr] * len(bb2s)
    else:
        colors = clr

    for i, bb2 in enumerate(bb2s):
        # draw a rectangle
        xmin = int(round(float(bb2[0])))
        xmax = int(round(float(bb2[1])))
        ymin = int(round(float(bb2[2])))
        ymax = int(round(float(bb2[3])))
        cc = colors[i]
        cv2.rectangle(
            img, (xmin, ymin), (xmax, ymax), cc, int(round(scale * 1)), lineType=16
        )
        if texts is not None and not rotated:
            # Place text in the center of the bounding box
            center_x = (xmin + xmax) // 2
            center_y = (ymin + ymax) // 2
            put_text(img, texts[i], scale=0.35, font_pt=(center_x, center_y), color=cc)

    if rotated:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if texts is not None:
            for i, bb2 in enumerate(bb2s):
                xmin = int(round(float(bb2[0])))
                xmax = int(round(float(bb2[1])))
                ymin = int(round(float(bb2[2])))
                ymax = int(round(float(bb2[3])))
                W = img.shape[1]  # Width of rotated image = Height of original
                # After 90° CW rotation: original (x,y) -> (H_orig - 1 - y, x)
                # Center of original box: ((xmin+xmax)/2, (ymin+ymax)/2)
                # Maps to rotated: (H_orig - 1 - (ymin+ymax)/2, (xmin+xmax)/2)
                # H_orig = W (width of rotated image)
                center_x = W - 1 - (ymin + ymax) // 2
                center_y = (xmin + xmax) // 2
                cc = colors[i]
                put_text(
                    img, texts[i], scale=0.35, font_pt=(center_x, center_y), color=cc
                )
    return img


def render_depth_patches(sdp_median, rotated, HH, WW):
    sdp_median = sdp_median[None]
    sdp_median = torch.nn.functional.interpolate(
        sdp_median, size=(HH, WW), mode="nearest"
    )
    sdp_median = sdp_median[0, 0]
    max_depth = 5.0
    min_depth = 0.1
    sdp_median = (sdp_median - min_depth) / (max_depth - min_depth)
    sdp_u8 = (sdp_median.clamp(0, 1).numpy() * 255).astype(np.uint8)
    sdp_img2 = cv2.applyColorMap(sdp_u8, cv2.COLORMAP_JET)
    if rotated:
        sdp_img2 = cv2.rotate(sdp_img2, cv2.ROTATE_90_CLOCKWISE)
    return sdp_img2
