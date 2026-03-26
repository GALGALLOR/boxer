# pyre-ignore-all-errors
import glob
import json
import os
import threading

import cv2
import numpy as np
import torch
from PIL import Image
from tw.camera import CameraTW
from tw.obb import ObbTW
from tw.pose import PoseTW
from tw.tensor_utils import pad_string, string2tensor

from loaders.base_loader import BaseLoader


def _read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _load_world_obbs(data_dir):
    """Load world-level OBBs and metadata from an extracted CA-1M directory."""
    world_path = os.path.join(data_dir, "world.gt", "instances.json")
    if not os.path.exists(world_path):
        raise ValueError(f"No world.gt/instances.json found in {data_dir}")

    bb3 = _read_json(world_path)
    id2inst = {}
    bb3_R = torch.stack([torch.tensor(bb["R"]) for bb in bb3])
    bb3_t = torch.stack([torch.tensor(bb["position"]) for bb in bb3])
    T_wo = PoseTW.from_Rt(bb3_R, bb3_t)
    bb3_sc = torch.stack([torch.tensor(bb["scale"]) for bb in bb3])
    xmin = -bb3_sc[:, 0] / 2
    xmax = bb3_sc[:, 0] / 2
    ymin = -bb3_sc[:, 1] / 2
    ymax = bb3_sc[:, 1] / 2
    zmin = -bb3_sc[:, 2] / 2
    zmax = bb3_sc[:, 2] / 2
    sz = torch.stack([xmin, xmax, ymin, ymax, zmin, zmax], dim=-1)

    ids = [bb["id"] for bb in bb3]
    for i, id_ in enumerate(ids):
        id2inst[id_] = i
    inst_ids = torch.tensor([id2inst[id_] for id_ in ids], dtype=torch.int64)

    category_names = [bb["category"] for bb in bb3]
    unique_categories = sorted(set(category_names))
    sem_id_to_name = {i: cat for i, cat in enumerate(unique_categories)}
    category_to_sem_id = {cat: i for i, cat in enumerate(unique_categories)}
    sem_ids = torch.tensor(
        [category_to_sem_id[bb["category"]] for bb in bb3], dtype=torch.int64
    )

    inst_id_to_caption = {}
    for bb in bb3:
        inst_id_to_caption[id2inst[bb["id"]]] = bb.get("caption", "")

    text = [string2tensor(pad_string(xx, max_len=128)) for xx in category_names]
    text = torch.stack(text)
    all_obbs = ObbTW.from_lmc(
        bb3_object=sz,
        T_world_object=T_wo,
        sem_id=sem_ids,
        inst_id=inst_ids,
        text=text,
    )

    return all_obbs, id2inst, sem_id_to_name, inst_id_to_caption


class CALoader(BaseLoader):
    def __init__(
        self,
        seq_name,
        start_frame=0,
        skip_frames=1,
        max_frames=10,
        resize=None,
        remove_structure=True,
        remove_large=True,
        min_dim=0.05,
        use_canny=True,
        num_samples=10000,
        filter_border_bbs=False,
        border_valid_ratio=0.95,
        frame_gt=False,
        bb2d_use_pseudo=True,
    ):
        seq_name = seq_name.strip("/")
        # Find the extracted data directory.
        from utils.demo_utils import SAMPLE_DATA_PATH

        sample = os.path.join(SAMPLE_DATA_PATH, seq_name)
        if os.path.exists(sample):
            out_dir = sample
        else:
            out_dir = os.path.expanduser(f"~/data/ca1m/{seq_name}")
        world_files = glob.glob(
            os.path.join(out_dir, "**/world.gt/instances.json"), recursive=True
        )
        if not world_files:
            raise FileNotFoundError(
                f"No extracted CA-1M data found in {out_dir}. "
                f"Run: python scripts/download_ca1m_sample.py --video-id {seq_name.split('-')[-1]}"
            )
        data_dir = os.path.dirname(os.path.dirname(world_files[0]))

        # Load world OBBs (lightweight, just JSON parsing).
        all_obbs, id2inst, sem_id_to_name, inst_id_to_caption = _load_world_obbs(
            data_dir
        )
        self.all_obbs = all_obbs
        self.id2inst = id2inst
        self.sem_id_to_name = sem_id_to_name
        self.sem_name_to_id = {v: k for k, v in sem_id_to_name.items()}
        self.inst_id_to_caption = inst_id_to_caption

        # Discover frame timestamps from .wide directories.
        wide_dirs = glob.glob(os.path.join(data_dir, "*.wide"))
        image_tags = sorted(
            {os.path.basename(d).split(".")[0] for d in wide_dirs}
        )
        image_tags = image_tags[start_frame:]
        image_tags = image_tags[::skip_frames]
        image_tags = image_tags[:max_frames]
        self.image_tags = image_tags

        self.data_dir = data_dir
        self.seq_name = seq_name
        self.local_root = out_dir
        self.length = len(image_tags)
        self.index = 0
        self.camera = "rgb"
        self.device_name = "ipad"
        self.resize = resize
        self.use_canny = use_canny
        self.num_samples = num_samples
        self.frame_gt = frame_gt
        self.remove_structure = remove_structure
        self.remove_large = remove_large
        self.max_dimension = 3.0
        self.min_dim = min_dim
        self.filter_border_bbs = filter_border_bbs
        self.border_valid_ratio = border_valid_ratio
        self.bb2d_use_pseudo = bb2d_use_pseudo
        self.num_border_bbs_filtered = 0

        # Find semantic IDs for floor and wall classes to filter out
        self.structure_sem_ids = []
        if remove_structure:
            for sem_id, name in self.sem_id_to_name.items():
                if name.lower() in ("floor", "wall"):
                    self.structure_sem_ids.append(sem_id)
                    print(f"==> filtering out: {name}: {sem_id}")

        # Timestamps derived from image tags (always available without loading images).
        self.timestamp_ns = torch.tensor(
            [int(tag) for tag in image_tags], dtype=torch.int64
        )

        # Cached bulk data (populated by load_all() for viewer use).
        self.rgb_images = None
        self.Ts_wc = None
        self.cams = None
        self.sdp_ws = None

        print(f"Found {self.length} frames in {data_dir}")

        # Prefetch: load next frame in background thread
        self._prefetch_result = None
        self._prefetch_thread = None
        self._start_prefetch()

    def load_all(self):
        """Pre-load all frames into memory (for viewer random access).

        Populates self.rgb_images, self.Ts_wc, self.cams, self.sdp_ws.
        """
        from tqdm import tqdm

        rgb_images = []
        Ts_wc = []
        cams = []
        sdp_ws = []
        for tag in tqdm(self.image_tags, desc="Loading frames"):
            frame = self._load_frame(tag)
            rgb_images.append(frame["image"])
            Ts_wc.append(frame["T_wc"].clone())
            cams.append(frame["cam"])
            sdp_ws.append(frame["sdp_w"].clone())
        self.rgb_images = rgb_images
        self.Ts_wc = torch.stack(Ts_wc)
        self.cams = torch.stack(cams)
        self.sdp_ws = torch.stack(sdp_ws)

    def __len__(self):
        return self.length

    def __iter__(self):
        return self

    def _load_frame(self, image_tag):
        """Load all data for a single frame from disk."""
        data_dir = self.data_dir

        # Load RGB image.
        image = np.array(
            Image.open(os.path.join(data_dir, image_tag + ".wide", "image.png"))
        )
        H, W = image.shape[:2]

        # Load camera extrinsics.
        RT = torch.tensor(
            _read_json(os.path.join(data_dir, image_tag + ".gt", "RT.json"))
        )
        T_wc = PoseTW.from_Rt(RT[:3, :3], RT[:3, 3])

        # Load camera intrinsics.
        K = torch.tensor(
            _read_json(
                os.path.join(data_dir, image_tag + ".wide", "image", "K.json")
            )
        )
        params = torch.tensor(
            [K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=torch.float32
        )
        cam = CameraTW.from_surreal(
            width=W, height=H, type_str="pinhole", params=params
        )

        # Load depth image.
        depth = np.array(
            Image.open(os.path.join(data_dir, image_tag + ".wide", "depth.png"))
        )
        Hd, Wd = depth.shape[:2]
        Kd = torch.tensor(
            _read_json(
                os.path.join(data_dir, image_tag + ".wide", "depth", "K.json")
            )
        )
        paramsd = torch.tensor(
            [Kd[0, 0], Kd[1, 1], Kd[0, 2], Kd[1, 2]], dtype=torch.float32
        )
        depth_cam = CameraTW.from_surreal(
            width=Wd, height=Hd, type_str="pinhole", params=paramsd
        )

        # Sample semi-dense points.
        depth_resize = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
        num_samples = self.num_samples
        if self.use_canny:
            try:
                weights = cv2.Canny(image, 30, 60)
                weights_flat = weights.ravel().astype(np.float64)
                weights_flat /= weights_flat.sum()
                idx = np.random.choice(
                    H * W, size=num_samples, replace=True, p=weights_flat
                )
                ys, xs = np.unravel_index(idx, (H, W))
            except ValueError:
                xs = np.random.randint(0, W, size=(num_samples))
                ys = np.random.randint(0, H, size=(num_samples))
        else:
            xs = np.random.randint(0, W, size=(num_samples))
            ys = np.random.randint(0, H, size=(num_samples))
        points = np.stack([xs, ys], axis=-1)
        points3, valid = cam.unproject(points[None])
        points3 = points3[0]
        valid = valid[0]
        points3 = points3[valid]
        hh = points[valid, 1].astype(int)
        ww = points[valid, 0].astype(int)
        zz = (
            torch.tensor(depth_resize[hh, ww].astype(np.float32)).float() / 1000.0
        )
        sdp_c = points3.reshape(-1, 3) * zz.reshape(-1, 1)
        sdp_w = T_wc * sdp_c
        if sdp_w.shape[0] < num_samples:
            num_pad = num_samples - sdp_w.shape[0]
            pad_vals = float("nan") * np.ones((num_pad, 3))
            sdp_w = np.concatenate([sdp_w, pad_vals], axis=0)
        sdp_w = torch.as_tensor(sdp_w).detach().clone()

        # Load per-frame 3D bounding boxes.
        bb3 = _read_json(
            os.path.join(data_dir, image_tag + ".wide", "instances.json")
        )
        visible_obbs = []
        frame_bb2ds = []
        frame_bb2d_inst_ids = []
        for bb in bb3:
            id_ = bb["id"]
            if id_ not in self.id2inst:
                continue
            inst_id = self.id2inst[id_]
            if self.frame_gt:
                bb_R = torch.tensor(bb["R"])
                bb_t = torch.tensor(bb["position"])
                T_co = PoseTW.from_Rt(bb_R[None], bb_t[None])
                T_wo_frame = T_wc @ T_co
                bb_sc = torch.tensor(bb["scale"])
                xmin = -bb_sc[0] / 2
                xmax = bb_sc[0] / 2
                ymin = -bb_sc[1] / 2
                ymax = bb_sc[1] / 2
                zmin = -bb_sc[2] / 2
                zmax = bb_sc[2] / 2
                sz_frame = torch.stack([xmin, xmax, ymin, ymax, zmin, zmax])[None]
                world_obb = self.all_obbs[inst_id]
                frame_obb = ObbTW.from_lmc(
                    bb3_object=sz_frame,
                    T_world_object=T_wo_frame,
                    sem_id=world_obb.sem_id[None],
                    inst_id=world_obb.inst_id[None],
                    text=world_obb.text[None],
                )
                visible_obbs.append(frame_obb[0])
            else:
                visible_obbs.append(self.all_obbs[inst_id].clone())
            if "box_2d_rend" in bb:
                x1, y1, x2, y2 = bb["box_2d_rend"]
                frame_bb2ds.append([x1, x2, y1, y2])
            elif "bbox" in bb:
                x_min, y_min, width, height = bb["bbox"]
                frame_bb2ds.append([x_min, x_min + width, y_min, y_min + height])
            else:
                frame_bb2ds.append([float("nan")] * 4)
            frame_bb2d_inst_ids.append(inst_id)
        visible_obbs = ObbTW.stack(visible_obbs).add_padding(512)
        if len(frame_bb2ds) > 0:
            frame_bb2ds = torch.tensor(frame_bb2ds, dtype=torch.float32)
            frame_bb2d_inst_ids = torch.tensor(
                frame_bb2d_inst_ids, dtype=torch.int64
            )
            num_pad = 512 - frame_bb2ds.shape[0]
            if num_pad > 0:
                pad_vals = torch.full((num_pad, 4), float("nan"))
                frame_bb2ds = torch.cat([frame_bb2ds, pad_vals], dim=0)
                pad_inst_ids = torch.full((num_pad,), -1, dtype=torch.int64)
                frame_bb2d_inst_ids = torch.cat(
                    [frame_bb2d_inst_ids, pad_inst_ids], dim=0
                )
        else:
            frame_bb2ds = torch.full((512, 4), float("nan"))
            frame_bb2d_inst_ids = torch.full((512,), -1, dtype=torch.int64)

        return {
            "image": image,
            "depth": depth,
            "cam": cam,
            "depth_cam": depth_cam,
            "T_wc": T_wc,
            "obbs": visible_obbs,
            "sdp_w": sdp_w,
            "bb2ds": frame_bb2ds,
            "bb2d_inst_ids": frame_bb2d_inst_ids,
            "timestamp_ns": int(image_tag),
        }

    def load(self, idx):
        """Load a single frame by index and return a datum dict."""
        image_tag = self.image_tags[idx]
        frame = self._load_frame(image_tag)

        datum = {}
        img = frame["image"]
        img_torch = torch.from_numpy(img).permute(2, 0, 1) / 255.0
        img_torch = img_torch[None]

        HH = img_torch.shape[2]
        WW = img_torch.shape[3]
        if self.resize is not None:
            if isinstance(self.resize, (tuple, list)):
                resizeH, resizeW = self.resize
            else:
                resizeH = self.resize
                resizeW = self.resize
            img_torch = torch.nn.functional.interpolate(
                img_torch,
                size=(resizeH, resizeW),
                mode="bilinear",
                align_corners=True,
            )
        datum["img0"] = img_torch.float()

        cam = frame["cam"]
        if self.resize is not None:
            cam = cam.scale((resizeW / WW, resizeH / HH))
        datum["cam0"] = cam.float()

        depth = frame["depth"]
        depth_torch = torch.from_numpy(depth.astype(np.float32))[None, None]
        HHd, WWd = depth_torch.shape[2:]
        if self.resize is not None:
            depth_torch = torch.nn.functional.interpolate(
                depth_torch,
                size=(resizeH, resizeW),
                mode="bilinear",
                align_corners=True,
            )
        datum["depth0"] = depth_torch

        depth_cam = frame["depth_cam"]
        if self.resize is not None:
            depth_cam = depth_cam.scale((resizeW / WWd, resizeH / HHd))
        datum["depth_cam0"] = depth_cam.float()

        T_wc = frame["T_wc"]
        T_wr = T_wc @ frame["cam"].T_camera_rig
        datum["T_world_rig0"] = T_wr.float()

        obbs = frame["obbs"].float().remove_padding()

        # Filter out floor/wall instances
        if len(self.structure_sem_ids) > 0:
            keep_mask = ~torch.isin(
                obbs.sem_id.squeeze(-1),
                torch.tensor(self.structure_sem_ids),
            )
            obbs = obbs[keep_mask]

        # Filter out large objects
        if self.remove_large and len(obbs) > 0:
            bb3 = obbs.bb3_object
            dims_x = bb3[:, 1] - bb3[:, 0]
            dims_y = bb3[:, 3] - bb3[:, 2]
            dims_z = bb3[:, 5] - bb3[:, 4]
            max_dims = torch.max(
                torch.stack([dims_x, dims_y, dims_z], dim=1), dim=1
            )[0]
            obbs = obbs[max_dims <= self.max_dimension]

        # Expand minimum object dimensions
        if self.min_dim > 0 and len(obbs) > 0:
            bb3 = obbs.bb3_object.clone()
            dims_x = bb3[:, 1] - bb3[:, 0]
            dims_y = bb3[:, 3] - bb3[:, 2]
            dims_z = bb3[:, 5] - bb3[:, 4]
            expand_x = (self.min_dim - dims_x).clamp(min=0) / 2
            expand_y = (self.min_dim - dims_y).clamp(min=0) / 2
            expand_z = (self.min_dim - dims_z).clamp(min=0) / 2
            bb3[:, 0] -= expand_x
            bb3[:, 1] += expand_x
            bb3[:, 2] -= expand_y
            bb3[:, 3] += expand_y
            bb3[:, 4] -= expand_z
            bb3[:, 5] += expand_z
            obbs.set_bb3_object(bb3)

        # Filter OBBs at image border
        if self.filter_border_bbs and len(obbs) > 0:
            num_before = len(obbs)
            obbs_batched = obbs[None]
            _, bb2_valid = obbs_batched.get_pseudo_bb2(
                datum["cam0"][None].unsqueeze(1),
                datum["T_world_rig0"][None].unsqueeze(1),
                num_samples_per_edge=50,
                valid_ratio=self.border_valid_ratio,
            )
            obbs = obbs[bb2_valid.squeeze(0).squeeze(0)]
            self.num_border_bbs_filtered += num_before - len(obbs)

        datum["obbs"] = obbs
        datum["sdp_w"] = frame["sdp_w"].float()
        datum["time_ns0"] = frame["timestamp_ns"]
        datum["rotated0"] = torch.tensor(False).reshape(1)
        datum["num_img"] = torch.tensor(1).reshape(1)

        # Compute or load 2D bounding boxes
        if self.bb2d_use_pseudo and len(obbs) > 0:
            obbs_batched = obbs[None]
            bb2d, bb2d_valid = obbs_batched.get_pseudo_bb2(
                datum["cam0"][None].unsqueeze(1),
                datum["T_world_rig0"][None].unsqueeze(1),
                num_samples_per_edge=10,
                valid_ratio=0.1667,
            )
            bb2d = bb2d.squeeze(0).squeeze(0)
            bb2d_valid = bb2d_valid.squeeze(0).squeeze(0)
            bb2d[~bb2d_valid] = float("nan")
            datum["bb2d0"] = bb2d
        elif len(obbs) > 0:
            bb2ds = frame["bb2ds"]
            bb2d_inst_ids = frame["bb2d_inst_ids"]
            obb_inst_ids = obbs.inst_id.squeeze(-1)
            valid_mask = bb2d_inst_ids >= 0
            bb2ds = bb2ds[valid_mask]
            bb2d_inst_ids = bb2d_inst_ids[valid_mask]
            keep_mask = torch.isin(bb2d_inst_ids, obb_inst_ids)
            datum["bb2d0"] = bb2ds[keep_mask]

        return datum

    def _start_prefetch(self):
        """Start prefetching the frame at self.index in a background thread."""
        if self.index >= self.length:
            return
        idx = self.index
        self._prefetch_result = None
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker, args=(idx,), daemon=True
        )
        self._prefetch_thread.start()

    def _prefetch_worker(self, idx):
        """Background worker that loads a single frame."""
        try:
            self._prefetch_result = self.load(idx=idx)
        except Exception as e:
            self._prefetch_result = e

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration

        # Wait for prefetched result
        if self._prefetch_thread is not None:
            self._prefetch_thread.join()
            out = self._prefetch_result
            self._prefetch_thread = None
            self._prefetch_result = None
        else:
            out = self.load(idx=self.index)

        if isinstance(out, Exception):
            raise out

        self.index += 1

        # Kick off prefetch for the next frame
        self._start_prefetch()

        return out
