"""
One-time export script for OWLv2.

Bundles everything into a single checkpoint:
    ~/data/owl/owlv2-base-patch16-ensemble.pt

Contains: traced text encoder, traced vision detector, config, tokenizer vocab+merges.

Requires: pip install transformers

Usage:
    conda run -n boxer python detectors/export_owl.py
"""

import io
import json
import os
import tempfile

import torch
import torch.nn as nn
from transformers import Owlv2ForObjectDetection, Owlv2Processor


MODEL_NAME = "google/owlv2-base-patch16-ensemble"
OUTPUT_DIR = os.path.expanduser("~/data/boxer")


class TextEncoderWrapper(nn.Module):
    """Wraps OWLv2 text encoder: input_ids, attention_mask -> text_embeds."""

    def __init__(self, model):
        super().__init__()
        self.text_model = model.owlv2.text_model
        self.text_projection = model.owlv2.text_projection

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = text_outputs[1]
        return self.text_projection(pooled)


class VisionDetectorWrapper(nn.Module):
    """Wraps OWLv2 vision encoder + detection heads.

    Takes pixel_values + pre-computed text embeddings, returns logits and pred_boxes.
    """

    def __init__(self, model):
        super().__init__()
        self.vision_model = model.owlv2.vision_model
        self.post_layernorm = model.owlv2.vision_model.post_layernorm
        self.layer_norm = model.layer_norm
        self.class_head = model.class_head
        self.box_head = model.box_head
        self.sigmoid = model.sigmoid
        # Pre-computed box bias for native resolution (no interpolation needed)
        self.register_buffer("box_bias", model.box_bias.clone())

    def forward(
        self,
        pixel_values: torch.Tensor,
        query_embeds: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Vision encoder
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        last_hidden_state = vision_outputs[0]

        # Post layernorm
        image_embeds = self.post_layernorm(last_hidden_state)

        # Merge CLS token with patch tokens
        class_token_out = torch.broadcast_to(image_embeds[:, :1, :], image_embeds[:, :-1].shape)
        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.layer_norm(image_embeds)

        # image_feats: [batch, num_patches, hidden_dim]
        image_feats = image_embeds

        # query_embeds: [num_queries, embed_dim] -> [1, num_queries, embed_dim]
        query_embeds_batched = query_embeds.unsqueeze(0)
        query_mask_batched = query_mask.unsqueeze(0)

        # Class prediction
        pred_logits, _ = self.class_head(image_feats, query_embeds_batched, query_mask_batched)

        # Box prediction
        pred_boxes = self.box_head(image_feats)
        pred_boxes = pred_boxes + self.box_bias
        pred_boxes = self.sigmoid(pred_boxes)

        return pred_logits, pred_boxes


def _traced_to_bytes(traced_module):
    """Serialize a traced module to bytes."""
    buf = io.BytesIO()
    torch.jit.save(traced_module, buf)
    return buf.getvalue()


def export():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading {MODEL_NAME}...")
    model = Owlv2ForObjectDetection.from_pretrained(MODEL_NAME).eval()
    processor = Owlv2Processor.from_pretrained(MODEL_NAME, use_fast=True)

    # --- Extract tokenizer data ---
    tok = processor.tokenizer
    # Save to temp dir to get vocab.json and merges.txt
    with tempfile.TemporaryDirectory() as tmpdir:
        tok.save_pretrained(tmpdir)
        with open(os.path.join(tmpdir, "vocab.json")) as f:
            tokenizer_vocab = json.load(f)
        with open(os.path.join(tmpdir, "merges.txt")) as f:
            tokenizer_merges = f.read()

    # --- Build config ---
    ip = processor.image_processor
    config = {
        "image_mean": ip.image_mean,
        "image_std": ip.image_std,
        "image_size": [ip.size["height"], ip.size["width"]],
        "max_seq_length": tok.model_max_length,
        "bos_token_id": tok.bos_token_id,
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": tok.pad_token_id,
        "vocab_size": tok.vocab_size,
    }

    # --- Trace text encoder ---
    print("Tracing text encoder...")
    text_encoder = TextEncoderWrapper(model).float().eval()
    dummy_ids = torch.randint(0, 1000, (2, config["max_seq_length"]), dtype=torch.long)
    dummy_mask = torch.ones_like(dummy_ids)
    with torch.no_grad():
        traced_text = torch.jit.trace(text_encoder, (dummy_ids, dummy_mask))

    # --- Trace vision detector ---
    print("Tracing vision detector...")
    vision_detector = VisionDetectorWrapper(model).float().eval()
    H, W = config["image_size"]
    dummy_pixels = torch.randn(1, 3, H, W)
    with torch.no_grad():
        dummy_text_embeds = text_encoder(dummy_ids, dummy_mask)
    dummy_query_mask = torch.ones(dummy_text_embeds.shape[0], dtype=torch.bool)

    with torch.no_grad():
        traced_vision = torch.jit.trace(
            vision_detector, (dummy_pixels, dummy_text_embeds, dummy_query_mask)
        )

    # --- Save single combined checkpoint ---
    checkpoint = {
        "text_encoder": _traced_to_bytes(traced_text),
        "vision_detector": _traced_to_bytes(traced_vision),
        "config": config,
        "tokenizer_vocab": tokenizer_vocab,
        "tokenizer_merges": tokenizer_merges,
    }
    ckpt_name = "owlv2-base-patch16-ensemble.pt"
    ckpt_path = os.path.join(OUTPUT_DIR, ckpt_name)
    torch.save(checkpoint, ckpt_path)
    size_mb = os.path.getsize(ckpt_path) / 1e6
    print(f"Saved {ckpt_path} ({size_mb:.1f} MB)")

    # --- Verify numerical accuracy ---
    print("\nVerifying numerical accuracy...")
    test_texts = ["a photo of a cat", "a photo of a dog"]
    test_image = torch.randint(0, 255, (1, 3, 480, 640), dtype=torch.float32)

    inputs = processor(
        text=test_texts, images=test_image, return_tensors="pt",
        size={"height": H, "width": W},
    )
    inputs["interpolate_pos_encoding"] = False
    with torch.no_grad():
        orig_outputs = model(**inputs)

    with torch.no_grad():
        new_text_embeds = traced_text(inputs["input_ids"], inputs["attention_mask"])
        new_query_mask = torch.ones(new_text_embeds.shape[0], dtype=torch.bool)
        new_logits, new_boxes = traced_vision(
            inputs["pixel_values"], new_text_embeds, new_query_mask,
        )

    logits_diff = (orig_outputs.logits - new_logits).abs().max().item()
    boxes_diff = (orig_outputs.pred_boxes - new_boxes).abs().max().item()
    print(f"  Max logits diff: {logits_diff:.2e}")
    print(f"  Max boxes diff:  {boxes_diff:.2e}")

    if logits_diff < 1e-4 and boxes_diff < 1e-4:
        print("  PASS: Traced models match original within tolerance.")
    else:
        print("  WARNING: Numerical differences exceed tolerance!")


if __name__ == "__main__":
    export()
