# pyre-unsafe

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
Data loader for ScanNet scenes with Scan2CAD 3D bounding box annotations.

Reads ScanNet's native format (per-frame color/depth/pose + Scan2CAD annotations)
and yields datums in the same format as OmniLoader for use with viz_omni3d.py.
"""

import json
import os
import time
from typing import Optional

import cv2
import numpy as np
import torch
from tw.camera import CameraTW
from tw.obb import ObbTW
from loaders.omni_loader import corners_to_obb
from tw.pose import PoseTW

# ShapeNet category ID to human-readable name
SHAPENET_CAT_MAP = {
    "03001627": "chair",
    "04379243": "table",
    "02933112": "cabinet",
    "02747177": "trash_bin",
    "02871439": "bookshelf",
    "03211117": "display",
    "04256520": "sofa",
    "02808440": "bathtub",
    "02818832": "bed",
    "03337140": "file_cabinet",
}


def _quat_to_rotation_matrix(q):
    """Convert quaternion [qw, qx, qy, qz] to 3x3 rotation matrix."""
    qw, qx, qy, qz = q
    return np.array(
        [
            [
                1 - 2 * (qy * qy + qz * qz),
                2 * (qx * qy - qz * qw),
                2 * (qx * qz + qy * qw),
            ],
            [
                2 * (qx * qy + qz * qw),
                1 - 2 * (qx * qx + qz * qz),
                2 * (qy * qz - qx * qw),
            ],
            [
                2 * (qx * qz - qy * qw),
                2 * (qy * qz + qx * qw),
                1 - 2 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )


def _make_M_from_tqs(t, q, s):
    """Build 4x4 transform M = T @ R @ S from translation, quaternion, scale.

    Matches the Scan2CAD make_M_from_tqs convention.

    Args:
        t: [3] translation
        q: [qw, qx, qy, qz] quaternion
        s: [3] scale

    Returns:
        4x4 numpy array (float64)
    """
    t = np.array(t, dtype=np.float64)
    s = np.array(s, dtype=np.float64)

    R = _quat_to_rotation_matrix(q)
    S = np.diag(s)

    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R @ S
    M[:3, 3] = t
    return M


from loaders.base_loader import BaseLoader


class ScanNetLoader(BaseLoader):
    """
    Data loader for ScanNet scenes with Scan2CAD annotations.

    Reads ScanNet's per-frame color/depth/pose files and Scan2CAD's
    full_annotations.json, yielding datums compatible with OmniLoader output.
    """

    def __init__(
        self,
        scene_dir: str,
        annotation_path: str,
        skip_frames: int = 1,
        max_frames: Optional[int] = None,
        start_frame: int = 1,
    ):
        # Allow short scene names like "scene0000_00" → sample_data/ or ~/data/scannet/
        scene_dir = os.path.expanduser(scene_dir)
        if not os.path.isabs(scene_dir) and not os.path.exists(scene_dir):
            from utils.demo_utils import SAMPLE_DATA_PATH
            sample = os.path.join(SAMPLE_DATA_PATH, scene_dir)
            legacy = os.path.expanduser(f"~/data/scannet/{scene_dir}")
            scene_dir = sample if os.path.exists(sample) else legacy
        self.scene_dir = scene_dir
        self.skip_frames = skip_frames
        self.resize = None  # Will be set by BoxerNetWrapper
        self.camera = "scannet"
        self.device_name = "ScanNet"

        # Extract scene_id from directory name (e.g. "scene0339_00")
        self.scene_id = os.path.basename(self.scene_dir.rstrip("/"))

        # Load intrinsics
        intrinsic_path = os.path.join(
            self.scene_dir, "frames", "intrinsic", "intrinsic_color.txt"
        )
        K = np.loadtxt(intrinsic_path)
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]

        # List and sort frame files
        color_dir = os.path.join(self.scene_dir, "frames", "color")
        frame_files = sorted(
            [
                f
                for f in os.listdir(color_dir)
                if f.endswith(".png") or f.endswith(".jpg")
            ],
            key=lambda x: int(os.path.splitext(x)[0]),
        )
        frame_ids = [os.path.splitext(f)[0] for f in frame_files]

        # Apply start frame offset and skip
        frame_ids = frame_ids[start_frame - 1 :: skip_frames]

        # Filter out frames with invalid poses
        valid_frame_ids = []
        for fid in frame_ids:
            pose_path = os.path.join(self.scene_dir, "frames", "pose", f"{fid}.txt")
            if os.path.exists(pose_path):
                pose = np.loadtxt(pose_path)
                if not (np.any(np.isinf(pose)) or np.any(np.isnan(pose))):
                    valid_frame_ids.append(fid)

        if max_frames is not None:
            valid_frame_ids = valid_frame_ids[:max_frames]

        self.frame_ids = valid_frame_ids
        self.length = len(self.frame_ids)
        self.index = 0

        # Load first valid pose to use as world origin
        first_pose_path = os.path.join(
            self.scene_dir, "frames", "pose", f"{self.frame_ids[0]}.txt"
        )
        first_pose = np.loadtxt(first_pose_path).astype(np.float64)
        self.world_offset = first_pose[:3, 3].copy()  # translation of first camera

        # Build sem_id_to_name mapping from ShapeNet categories
        self.sem_id_to_name = {}
        self.sem_name_to_id = {}
        for i, (shapenet_id, name) in enumerate(SHAPENET_CAT_MAP.items()):
            self.sem_id_to_name[i] = name
            self.sem_name_to_id[name] = i
        # Also store the shapenet mapping for lookup
        self._shapenet_to_sem_id = {
            shapenet_id: i for i, shapenet_id in enumerate(SHAPENET_CAT_MAP.keys())
        }

        # Load annotations and precompute scan-space box corners
        self._load_annotations(annotation_path)

        print(
            f"ScanNetLoader: {self.scene_id}, {self.length} frames, "
            f"{len(self.scan_corners)} 3D boxes"
        )

    def _load_annotations(self, annotation_path: str):
        """Load Scan2CAD annotations and precompute scan-space box corners.

        Follows the Scan2CAD calc_Mbbox transform chain:
        1. Build scan TRS matrix and INVERT it (scan TRS maps scan→world,
           we need world→scan to work in ScanNet's coordinate system)
        2. For each model: build pose WITHOUT scale (rotation + translation only)
        3. Scale is applied separately to bbox half-extents and center offset
        4. Compose: T_scan_object = inv(T_world_scan) @ T_world_object @ center_offset
        5. Corners (symmetric, scaled) are transformed by T_scan_object
        """
        annotation_path = os.path.expanduser(annotation_path)
        with open(annotation_path, "r") as f:
            all_annotations = json.load(f)

        # Find the annotation entry for this scene
        scene_ann = None
        for ann in all_annotations:
            if ann["id_scan"] == self.scene_id:
                scene_ann = ann
                break

        self.scan_corners = []  # list of (8, 3) np arrays in scan/ScanNet space
        self.box_cat_ids = []  # semantic category IDs
        self.box_cat_names = []  # category name strings

        if scene_ann is None:
            print(f"Warning: No annotations found for {self.scene_id}")
            return

        # Scan TRS: maps scan→world. Invert to get world→scan (= ScanNet space).
        scan_trs = scene_ann["trs"]
        T_ws = _make_M_from_tqs(
            scan_trs["translation"], scan_trs["rotation"], scan_trs["scale"]
        )
        T_sw = np.linalg.inv(T_ws)

        for model in scene_ann["aligned_models"]:
            model_trs = model["trs"]
            scale = np.array(model_trs["scale"], dtype=np.float64)

            # Skip degenerate models
            if scale.min() < 1e-3:
                continue

            # Model pose: rotation + translation, NO scale
            T_wo = _make_M_from_tqs(
                model_trs["translation"], model_trs["rotation"], [1.0, 1.0, 1.0]
            )

            # Center offset (scaled by object scale)
            center = np.array(model["center"], dtype=np.float64)
            mat_off = np.eye(4, dtype=np.float64)
            mat_off[:3, 3] = center * scale

            # Combined: scan-from-object = inv(scan_trs) @ model_pose @ center_offset
            T_so = T_sw @ T_wo @ mat_off

            # Build corners: symmetric around origin, scaled by half-extents * scale
            half_extents = np.array(model["bbox"], dtype=np.float64) * scale
            signs = np.array(
                [
                    [-1, -1, -1],
                    [+1, -1, -1],
                    [+1, +1, -1],
                    [-1, +1, -1],
                    [-1, -1, +1],
                    [+1, -1, +1],
                    [+1, +1, +1],
                    [-1, +1, +1],
                ],
                dtype=np.float64,
            )
            corners_local = signs * half_extents  # (8, 3)

            # Transform to scan space
            corners_h = np.hstack([corners_local, np.ones((8, 1), dtype=np.float64)])
            corners_scan = (T_so @ corners_h.T).T[:, :3]

            # Recenter around first camera pose
            corners_scan -= self.world_offset

            self.scan_corners.append(corners_scan.astype(np.float32))

            # Category mapping
            catid = model.get("catid_cad", "")
            cat_name = SHAPENET_CAT_MAP.get(catid, "unknown")
            sem_id = self._shapenet_to_sem_id.get(catid, -1)
            if sem_id == -1:
                sem_id = len(self.sem_id_to_name)
                self.sem_id_to_name[sem_id] = cat_name
            self.box_cat_ids.append(sem_id)
            self.box_cat_names.append(cat_name)

    def __len__(self):
        return self.length

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration

        frame_id = self.frame_ids[self.index]
        datum = {}
        _debug = os.environ.get("DEBUG_VIZ", "0") == "1"

        # Load color image
        if _debug:
            _t0 = time.perf_counter()
        color_path = os.path.join(self.scene_dir, "frames", "color", f"{frame_id}.png")
        if not os.path.exists(color_path):
            # Try .jpg
            color_path = os.path.join(
                self.scene_dir, "frames", "color", f"{frame_id}.jpg"
            )
        img_bgr = cv2.imread(color_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        HH, WW = img_rgb.shape[:2]

        # Resize if requested (e.g. by BoxerNetWrapper)
        if self.resize is not None:
            resizeH = resizeW = self.resize
            scale_x = resizeW / WW
            scale_y = resizeH / HH
        else:
            resizeH, resizeW = HH, WW
            scale_x = scale_y = 1.0

        fx = self.fx * scale_x
        fy = self.fy * scale_y
        cx = self.cx * scale_x
        cy = self.cy * scale_y

        if self.resize is not None:
            img_rgb = cv2.resize(
                img_rgb, (resizeW, resizeH), interpolation=cv2.INTER_LINEAR
            )
            img_bgr = cv2.resize(
                img_bgr, (resizeW, resizeH), interpolation=cv2.INTER_LINEAR
            )

        if _debug:
            _t1 = time.perf_counter()

        # Convert to torch [1, 3, H, W] normalized to [0, 1]
        img_torch = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        datum["img0"] = img_torch[None]
        # Keep BGR uint8 for viz (avoids torch2cv2 round-trip)
        datum["img_bgr0"] = img_bgr

        if _debug:
            _t2 = time.perf_counter()

        # Load depth (uint16, mm) and resize to match output resolution
        depth_path = os.path.join(self.scene_dir, "frames", "depth", f"{frame_id}.png")
        depth_np = None
        if os.path.exists(depth_path):
            depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_raw is not None:
                depth_np = depth_raw.astype(np.float32) / 1000.0  # mm to meters
                # Resize depth to match output resolution
                if depth_np.shape[:2] != (resizeH, resizeW):
                    depth_np = cv2.resize(
                        depth_np, (resizeW, resizeH), interpolation=cv2.INTER_NEAREST
                    )

        if depth_np is not None:
            datum["depth0"] = torch.from_numpy(depth_np).float()[None, None]
        else:
            datum["depth0"] = torch.zeros(1, 1, resizeH, resizeW, dtype=torch.float32)

        if _debug:
            _t3 = time.perf_counter()

        # Build pinhole camera
        T_cam_rig = torch.tensor(
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float32
        )
        cam_data = torch.tensor(
            # Make valid_radius 2x larger for ScanNet to be less restrictive.
            [resizeW, resizeH, fx, fy, cx, cy, -1, 1e-3, resizeW, resizeH],
            dtype=torch.float32,
        )
        cam_data = torch.cat([cam_data, T_cam_rig])
        cam = CameraTW(cam_data)
        datum["cam0"] = cam.float()
        datum["depth_cam0"] = cam.float()

        # Load camera pose (4x4 camera-to-world) and recenter
        pose_path = os.path.join(self.scene_dir, "frames", "pose", f"{frame_id}.txt")
        T_world_cam = np.loadtxt(pose_path).astype(np.float64)
        T_world_cam[:3, 3] -= self.world_offset  # recenter around first pose
        T_world_cam = T_world_cam.astype(np.float32)
        T_cam_world = np.linalg.inv(T_world_cam).astype(np.float32)

        # Transform scan-space box corners to camera space and create OBBs
        # (scan space = ScanNet world, same coordinate system as camera poses)
        obb_list = []
        for i, corners_scan in enumerate(self.scan_corners):
            corners_h = np.hstack([corners_scan, np.ones((8, 1), dtype=np.float32)])
            corners_cam = (T_cam_world @ corners_h.T).T[:, :3]

            # Skip boxes behind camera (center z <= 0)
            center_z = corners_cam[:, 2].mean()
            if center_z <= 0:
                continue

            obb = corners_to_obb(
                corners_cam, self.box_cat_ids[i], self.box_cat_names[i]
            )
            obb_list.append(obb)

        if len(obb_list) > 0:
            obbs = ObbTW(torch.stack([obb._data for obb in obb_list]))
        else:
            obbs = ObbTW(torch.zeros(0, 165))

        datum["obbs"] = obbs

        # T_world_rig from recentered camera pose
        R_flat = T_world_cam[:3, :3].flatten()  # row-major
        t_vec = T_world_cam[:3, 3]
        T_wr_data = torch.tensor([*R_flat, *t_vec], dtype=torch.float32)
        datum["T_world_rig0"] = PoseTW(T_wr_data)

        if _debug:
            _t4 = time.perf_counter()

        # Semi-dense points from depth (uniform subsampling)
        # Fast numpy-only pinhole unprojection. CameraTW.unproject() adds ~14ms
        # of overhead per frame (tensor alloc, model dispatch, in_image/in_radius
        # checks) that dominates for 10k points. Direct math: 0.7ms.
        num_samples = 10000
        if depth_np is not None:
            # Subsample the depth image on a grid, then filter valid
            dh, dw = depth_np.shape
            step = max(1, int(np.sqrt(dh * dw / (num_samples * 2))))
            ys_grid, xs_grid = np.mgrid[0:dh:step, 0:dw:step]
            ys_flat = ys_grid.ravel()
            xs_flat = xs_grid.ravel()
            zz = depth_np[ys_flat, xs_flat]
            valid_mask = zz > 0
            ys_v = ys_flat[valid_mask]
            xs_v = xs_flat[valid_mask]
            zz_v = zz[valid_mask]

            if len(ys_v) > num_samples:
                idx = np.random.choice(len(ys_v), size=num_samples, replace=False)
                ys_v, xs_v, zz_v = ys_v[idx], xs_v[idx], zz_v[idx]

            if len(ys_v) > 0:
                # Pinhole unproject: ray = [(x-cx)/fx, (y-cy)/fy, 1] * depth
                x3d = (xs_v.astype(np.float32) - cx) / fx * zz_v
                y3d = (ys_v.astype(np.float32) - cy) / fy * zz_v
                sdp_c = np.stack([x3d, y3d, zz_v], axis=-1)

                # Transform to world space: sdp_w = R @ sdp_c + t
                R = T_world_cam[:3, :3]
                t = T_world_cam[:3, 3]
                sdp_w_np = (sdp_c @ R.T) + t

                sdp_w = torch.from_numpy(sdp_w_np)
                if sdp_w.shape[0] < num_samples:
                    num_pad = num_samples - sdp_w.shape[0]
                    pad_vals = torch.full(
                        (num_pad, 3), float("nan"), dtype=torch.float32
                    )
                    sdp_w = torch.cat([sdp_w, pad_vals], dim=0)
                datum["sdp_w"] = sdp_w.float()
            else:
                datum["sdp_w"] = torch.zeros(0, 3, dtype=torch.float32)
        else:
            datum["sdp_w"] = torch.zeros(0, 3, dtype=torch.float32)

        if _debug:
            _t5 = time.perf_counter()

        # Metadata
        datum["time_ns0"] = int(frame_id)
        datum["rotated0"] = torch.tensor(False).reshape(1)
        datum["num_img"] = torch.tensor(1).reshape(1)
        datum["bb2d0"] = torch.zeros(0, 4, dtype=torch.float32)
        datum["gt_labels"] = []

        if _debug:
            _t6 = time.perf_counter()
            print(f"    [loader] imread: {(_t1-_t0)*1000:.1f}ms  to_torch: {(_t2-_t1)*1000:.1f}ms  depth: {(_t3-_t2)*1000:.1f}ms  cam+pose: {(_t4-_t3)*1000:.1f}ms  sdp: {(_t5-_t4)*1000:.1f}ms  total: {(_t6-_t0)*1000:.1f}ms", flush=True)

        self.index += 1
        return datum
