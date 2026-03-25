# pyre-unsafe
import io
import os

import torch
import torch.nn.functional as F
from detectors.clip_tokenizer import CLIPTokenizer

_CKPT_PATH = os.path.expanduser("~/data/boxer/owlv2-base-patch16-ensemble.pt")


class TextEmbedder:
    """Compute normalized text embeddings using the OWLv2 CLIP text encoder.

    Uses the same traced text encoder from the OWLv2 checkpoint — no additional
    dependencies beyond torch.
    """

    def __init__(self):
        if not os.path.exists(_CKPT_PATH):
            raise FileNotFoundError(
                f"OWLv2 checkpoint not found at {_CKPT_PATH}. "
                "Run 'python detectors/export_owl.py' first."
            )
        checkpoint = torch.load(_CKPT_PATH, map_location="cpu", weights_only=False)
        config = checkpoint["config"]

        self.text_encoder = torch.jit.load(
            io.BytesIO(checkpoint["text_encoder"]), map_location="cpu"
        )
        self.text_encoder.eval()

        self.tokenizer = CLIPTokenizer(
            vocab=checkpoint["tokenizer_vocab"],
            merges=checkpoint["tokenizer_merges"],
            max_length=config["max_seq_length"],
        )

    @torch.no_grad()
    def forward(self, texts: list[str]) -> torch.Tensor:
        """Encode a list of strings into L2-normalized embeddings.

        Args:
            texts: List of strings to encode.

        Returns:
            Tensor of shape (len(texts), embed_dim) with L2-normalized rows.
        """
        tokens = self.tokenizer(texts)
        embeds = self.text_encoder(tokens["input_ids"], tokens["attention_mask"])
        return F.normalize(embeds, dim=-1)
