# pyre-unsafe
import io
import os

import torch
import torch.nn.functional as F
from detectors.clip_tokenizer import CLIPTokenizer
from utils.taxonomy import load_text_labels

DEFAULT_TEXT_LABELS = load_text_labels("lvisplus")

_CKPT_PATH = os.path.expanduser("~/data/boxer/owlv2-base-patch16-ensemble.pt")


class OwlWrapper(torch.nn.Module):
    """
    Runs OWLv2 open-set 2D BB detector using JIT-traced models.
    No transformers dependency at runtime.

    Text embeddings are computed once at init and cached.
    Use set_text_prompts() to change prompts without re-creating the wrapper.
    """

    def __init__(self, device="cuda", text_prompts=None, min_confidence=0.2):
        super().__init__()

        if text_prompts is None:
            text_prompts = DEFAULT_TEXT_LABELS

        if device == "cuda":
            assert torch.cuda.is_available()
        elif device == "mps":
            assert torch.backends.mps.is_available()
        else:
            device = "cpu"

        # Load combined checkpoint
        if not os.path.exists(_CKPT_PATH):
            raise FileNotFoundError(
                f"OWLv2 checkpoint not found at {_CKPT_PATH}. "
                "Run 'python detectors/export_owl.py' first (requires transformers)."
            )
        checkpoint = torch.load(_CKPT_PATH, map_location="cpu", weights_only=False)

        config = checkpoint["config"]
        self.image_mean = torch.tensor(config["image_mean"]).view(1, 3, 1, 1)
        self.image_std = torch.tensor(config["image_std"]).view(1, 3, 1, 1)
        self.native_size = tuple(config["image_size"])  # (960, 960)

        # Load traced models from bytes
        # Text encoder always on CPU (runs once at init, not perf-critical)
        self.text_encoder = torch.jit.load(io.BytesIO(checkpoint["text_encoder"]), map_location="cpu")
        self.text_encoder.eval()

        # Vision detector on target device (load on CPU first for MPS compat)
        vis_map_loc = "cpu" if device == "mps" else device
        self.vision_detector = torch.jit.load(
            io.BytesIO(checkpoint["vision_detector"]), map_location=vis_map_loc
        )
        self.vision_detector.eval()
        if device == "mps":
            self.vision_detector = self.vision_detector.to(device)

        # Load tokenizer from checkpoint data
        self.tokenizer = CLIPTokenizer(
            vocab=checkpoint["tokenizer_vocab"],
            merges=checkpoint["tokenizer_merges"],
            max_length=config["max_seq_length"],
        )

        self.device = device
        self.text_prompts = text_prompts
        self.min_confidence = min_confidence

        # Pre-compute and cache text embeddings
        self.text_embeddings = self._encode_text(text_prompts)
        self.query_mask = torch.ones(len(text_prompts), dtype=torch.bool, device=device)

        print(f"Loaded OWLv2 (traced) on {device} with {len(text_prompts)} text prompts")

        # Warmup
        self._warmup()

    def _encode_text(self, prompts):
        """Tokenize and encode text prompts through the traced text encoder (runs on CPU)."""
        tokens = self.tokenizer(prompts)
        input_ids = tokens["input_ids"]  # CPU
        attention_mask = tokens["attention_mask"]  # CPU
        with torch.no_grad():
            embeds = self.text_encoder(input_ids, attention_mask)
        return embeds.to(self.device)

    def set_text_prompts(self, prompts):
        """Update text prompts and re-compute cached embeddings."""
        self.text_prompts = prompts
        self.text_embeddings = self._encode_text(prompts)
        self.query_mask = torch.ones(len(prompts), dtype=torch.bool, device=self.device)

    def _warmup(self, steps=3):
        """Warmup the vision model with dummy inference."""
        H, W = self.native_size
        dummy = torch.zeros(1, 3, H, W, device=self.device)
        with torch.no_grad():
            for _ in range(steps):
                self.vision_detector(dummy, self.text_embeddings, self.query_mask)

    @torch.no_grad()
    def forward(self, image_torch, rotated=False, resize_to_HW=(906, 906)):
        assert len(image_torch.shape) == 4, "input image should be 4D tensor"
        assert image_torch.shape[0] == 1, "only batch size 1 is supported"
        if (
            image_torch.max() < 1.01
            or image_torch.max() > 255.0
            or image_torch.min() < 0.0
        ):
            print("warning: input image should be in [0, 255] as a float")

        input_image = image_torch.clone()
        if rotated:
            input_image = torch.rot90(input_image, k=3, dims=(2, 3))  # 90 CW
        HH, WW = input_image.shape[2], input_image.shape[3]

        # Preprocess: resize to native model resolution, normalize
        interp_mode = "bilinear" if self.device == "mps" else "bicubic"
        pixel_values = F.interpolate(
            input_image, size=self.native_size, mode=interp_mode, align_corners=False,
        )
        pixel_values = pixel_values / 255.0
        mean = self.image_mean.to(pixel_values.device)
        std = self.image_std.to(pixel_values.device)
        pixel_values = (pixel_values - mean) / std
        pixel_values = pixel_values.to(self.device)

        # Forward pass (vision + detection)
        logits, pred_boxes = self.vision_detector(
            pixel_values, self.text_embeddings, self.query_mask,
        )

        # Postprocess: sigmoid, threshold, convert boxes
        scores_all, labels_all = torch.max(logits[0], dim=-1)  # [num_patches]
        scores_all = torch.sigmoid(scores_all)

        keep = scores_all > self.min_confidence
        scores = scores_all[keep].cpu()
        labels = labels_all[keep].cpu()
        boxes_cxcywh = pred_boxes[0, keep]  # [N, 4] normalized cxcywh

        empty_return = torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0), None

        if len(boxes_cxcywh) == 0:
            return empty_return

        # Convert cxcywh -> xyxy, scale to original image size
        cx, cy, w, h = boxes_cxcywh.unbind(-1)
        x1 = (cx - w / 2) * WW
        y1 = (cy - h / 2) * HH
        x2 = (cx + w / 2) * WW
        y2 = (cy + h / 2) * HH
        boxes = torch.stack([x1, y1, x2, y2], dim=-1).cpu()

        # Filter out very large or small boxes (false positives)
        too_big = (x2 - x1 > 0.9 * WW) | (y2 - y1 > 0.9 * HH)
        too_small = (x2 - x1 < 0.05 * WW) | (y2 - y1 < 0.05 * HH)
        keep = ~(too_big | too_small).cpu()
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        if len(boxes) == 0:
            return empty_return

        # Convert x1, y1, x2, y2 -> x1, x2, y1, y2 convention
        boxes = boxes[:, [0, 2, 1, 3]]

        if rotated:
            # Rotate boxes back by 90 degrees counter-clockwise
            x1, x2, y1, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            new_x1 = y1
            new_x2 = y2
            new_y1 = WW - x2
            new_y2 = WW - x1
            boxes = torch.stack([new_x1, new_x2, new_y1, new_y2], dim=-1)

        return boxes, scores, labels, None
