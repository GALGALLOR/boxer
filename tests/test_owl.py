#!/usr/bin/env python3

# pyre-unsafe

"""
Tests that the traced OWLv2 wrapper produces outputs matching the
original transformers-based pipeline.

Requires: pip install transformers  (only for this test)
"""

import unittest

import torch
from detectors.clip_tokenizer import CLIPTokenizer
from detectors.owl_wrapper import OwlWrapper
from utils.taxonomy import load_text_labels


def _has_transformers():
    try:
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


@unittest.skipUnless(_has_transformers(), "transformers not installed")
class TestOwlMatchesTransformers(unittest.TestCase):
    """Compare traced OWLv2 outputs against the original transformers model."""

    @classmethod
    def setUpClass(cls):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        cls.model_name = "google/owlv2-base-patch16-ensemble"
        cls.processor = Owlv2Processor.from_pretrained(cls.model_name, use_fast=True)
        cls.hf_model = Owlv2ForObjectDetection.from_pretrained(cls.model_name).eval()
        cls.traced_wrapper = OwlWrapper("cpu", text_prompts=["cat", "dog"], min_confidence=0.01)

    def _run_hf_pipeline(self, image, text_prompts):
        """Run the original transformers pipeline and return raw logits + boxes."""
        inputs = self.processor(
            text=text_prompts,
            images=image,
            return_tensors="pt",
            size={"height": 960, "width": 960},
        )
        # Use interpolate_pos_encoding=False to match traced model (native 960x960)
        inputs["interpolate_pos_encoding"] = False
        with torch.no_grad():
            outputs = self.hf_model(**inputs)
        return outputs.logits, outputs.pred_boxes

    def test_tokenizer_matches(self):
        """Verify our CLIP tokenizer matches the transformers tokenizer."""
        ref_tok = self.processor.tokenizer
        my_tok = self.traced_wrapper.tokenizer

        labels = load_text_labels("lvisplus")
        ref = ref_tok(labels, return_tensors="pt", padding="max_length", max_length=16, truncation=True)
        mine = my_tok(labels)

        self.assertTrue(
            (ref["input_ids"] == mine["input_ids"]).all(),
            "Tokenizer input_ids mismatch",
        )
        self.assertTrue(
            (ref["attention_mask"] == mine["attention_mask"]).all(),
            "Tokenizer attention_mask mismatch",
        )

    def test_text_embeddings_match(self):
        """Verify traced text encoder produces same embeddings as transformers."""
        prompts = ["a photo of a cat", "chair", "table"]
        ref_tok = self.processor.tokenizer
        ref_inputs = ref_tok(
            prompts, return_tensors="pt", padding="max_length", max_length=16, truncation=True
        )

        # Reference: transformers text encoder
        with torch.no_grad():
            text_out = self.hf_model.owlv2.text_model(
                input_ids=ref_inputs["input_ids"],
                attention_mask=ref_inputs["attention_mask"],
            )
            ref_embeds = self.hf_model.owlv2.text_projection(text_out[1])

        # Ours: traced text encoder
        wrapper = OwlWrapper("cpu", text_prompts=prompts, min_confidence=0.1)
        our_embeds = wrapper.text_embeddings

        max_diff = (ref_embeds - our_embeds).abs().max().item()
        self.assertLess(max_diff, 1e-4, f"Text embedding max diff {max_diff:.2e} exceeds tolerance")

    def test_vision_logits_match(self):
        """Verify traced vision detector logits match transformers on same preprocessed input."""
        prompts = ["cat", "dog"]
        torch.manual_seed(42)
        test_image = torch.randint(0, 255, (1, 3, 480, 640), dtype=torch.float32)

        # Get HF preprocessed input and run HF model
        inputs = self.processor(
            text=prompts,
            images=test_image,
            return_tensors="pt",
            size={"height": 960, "width": 960},
        )
        inputs["interpolate_pos_encoding"] = False
        with torch.no_grad():
            hf_out = self.hf_model(**inputs)

        # Run traced model with SAME preprocessed pixels (to isolate model diff from preproc diff)
        wrapper = OwlWrapper("cpu", text_prompts=prompts, min_confidence=0.01)
        with torch.no_grad():
            traced_logits, traced_boxes = wrapper.vision_detector(
                inputs["pixel_values"],
                wrapper.text_embeddings,
                wrapper.query_mask,
            )

        logits_diff = (hf_out.logits - traced_logits).abs().max().item()
        boxes_diff = (hf_out.pred_boxes - traced_boxes).abs().max().item()

        self.assertLess(logits_diff, 1e-4, f"Logits max diff {logits_diff:.2e}")
        self.assertLess(boxes_diff, 1e-4, f"Boxes max diff {boxes_diff:.2e}")

    def test_detection_output_format(self):
        """Verify forward() returns correct types and shapes."""
        wrapper = OwlWrapper("cpu", text_prompts=["cat", "dog"], min_confidence=0.01)
        img = torch.rand(1, 3, 480, 640) * 255
        boxes, scores, labels, masks = wrapper.forward(img)

        self.assertIsInstance(boxes, torch.Tensor)
        self.assertIsInstance(scores, torch.Tensor)
        self.assertIsInstance(labels, torch.Tensor)
        self.assertIsNone(masks)
        self.assertEqual(boxes.ndim, 2)
        self.assertEqual(boxes.shape[1], 4)
        self.assertEqual(scores.shape[0], boxes.shape[0])
        self.assertEqual(labels.shape[0], boxes.shape[0])

    def test_set_text_prompts(self):
        """Verify text prompts can be changed dynamically."""
        wrapper = OwlWrapper("cpu", text_prompts=["cat"], min_confidence=0.01)
        self.assertEqual(wrapper.text_embeddings.shape[0], 1)

        wrapper.set_text_prompts(["cat", "dog", "bird"])
        self.assertEqual(wrapper.text_embeddings.shape[0], 3)
        self.assertEqual(len(wrapper.text_prompts), 3)
        self.assertEqual(wrapper.query_mask.shape[0], 3)

    def test_rotated_input(self):
        """Verify rotated input doesn't crash and returns valid output."""
        wrapper = OwlWrapper("cpu", text_prompts=["cat"], min_confidence=0.01)
        img = torch.rand(1, 3, 480, 640) * 255
        boxes, scores, labels, masks = wrapper.forward(img, rotated=True)

        self.assertIsInstance(boxes, torch.Tensor)
        self.assertEqual(boxes.shape[1], 4) if len(boxes) > 0 else None

    def test_box_coordinate_convention(self):
        """Verify boxes use x1, x2, y1, y2 convention (not x1, y1, x2, y2)."""
        # Use a very low threshold to get some detections even on noise
        wrapper = OwlWrapper("cpu", text_prompts=["cat", "dog"], min_confidence=0.001)
        torch.manual_seed(123)
        img = torch.rand(1, 3, 480, 640) * 255
        boxes, scores, labels, _ = wrapper.forward(img)

        if len(boxes) > 0:
            # Convention: x1, x2, y1, y2 where x1 <= x2 and y1 <= y2
            x1, x2, y1, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            self.assertTrue((x1 <= x2).all(), "x1 > x2 in some boxes")
            self.assertTrue((y1 <= y2).all(), "y1 > y2 in some boxes")


class TestOwlWrapperStandalone(unittest.TestCase):
    """Tests that don't require transformers."""

    def test_no_transformers_imported(self):
        """Verify the wrapper doesn't import transformers at runtime."""
        import sys

        # Clear any cached imports
        mods_before = set(sys.modules.keys())
        _ = OwlWrapper("cpu", text_prompts=["test"], min_confidence=0.5)
        mods_after = set(sys.modules.keys())
        new_mods = mods_after - mods_before
        self.assertNotIn("transformers", new_mods)

    def test_missing_models_error(self):
        """Verify helpful error when traced models are missing."""
        import detectors.owl_wrapper as ow

        original = ow._CKPT_PATH
        ow._CKPT_PATH = "/nonexistent/path/model.pt"
        try:
            with self.assertRaises(FileNotFoundError) as ctx:
                OwlWrapper("cpu", text_prompts=["test"])
            self.assertIn("README", str(ctx.exception))
        finally:
            ow._CKPT_PATH = original


if __name__ == "__main__":
    unittest.main()
