# pyre-unsafe
import torch
from sentence_transformers import SentenceTransformer


class SentenceTransformerWrapper:
    """Thin wrapper around sentence-transformers for text embedding."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def forward(self, texts: list[str]) -> torch.Tensor:
        """Encode a list of strings into normalized embeddings.

        Args:
            texts: List of strings to encode.

        Returns:
            Tensor of shape (len(texts), embedding_dim) with L2-normalized rows.
        """
        embeddings = self.model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        return embeddings
