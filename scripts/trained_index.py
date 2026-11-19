"""TrainedOptionIndex: OptionIndex using a trained HDC encoder.

Wraps the two-stage ranking but with learned text encodings instead of SimHash.
Produces the same API as OptionIndex: fit, rank.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from hd_laya.index import OptionIndex
from scripts.train_hdc_encoder import HdcEncoder


class TrainedOptionIndex:
    """OptionIndex with trained HDC encoder (cosine similarity instead of Hamming)."""

    def __init__(self, options, encoder, dims=2048):
        self.options = list(options)
        self.encoder = encoder
        self.dims = dims
        self.dims_full = dims
        self.seed = "trained-hdc"
        self.ngram = 3

        # Encode all options
        self._hvs = []
        for opt in self.options:
            hv = self.encode_text(opt)
            self._hvs.append(hv)
        # (n, dims) float32
        self._hvs = torch.stack(self._hvs).cpu().numpy()

    def encode_text(self, text):
        from scripts.train_hdc_encoder import text_to_features
        X = torch.tensor(text_to_features([text], dims=self.dims), device="cpu")
        with torch.no_grad():
            hv = self.encoder(X)
        return F.normalize(hv, dim=1)[0]

    def fit(cls, options, encoder, **kwargs):
        return cls(options, encoder)

    @classmethod
    def from_file(cls, options, model_path, **kwargs):
        encoder = HdcEncoder(dims=2048, hidden=2048)
        state = torch.load(model_path, map_location="cpu")["model"]
        encoder.load_state_dict(state)
        encoder.eval()
        return cls(options, encoder)

    def rank(self, query_text, k=1024, M=64):
        """Two-stage ranking: coarse scan + fine re-rank."""
        n = len(self.options)
        if n == 0:
            return [], np.zeros(0), {"final_top1": None, "margin": 0.0}

        query_hv = self.encode_text(query_text).cpu().numpy()

        # Stage 1: compute all similarities, pick top-M candidates
        sims = self._hvs @ query_hv  # (n,)
        M = min(int(M), n)
        cand_idx = np.argsort(sims)[::-1][:M]

        # Stage 2: fine re-rank of candidates
        cand_hvs = self._hvs[cand_idx]
        cand_sims = cand_hvs @ query_hv
        order2 = np.argsort(cand_sims)[::-1]
        final_idx = cand_idx[order2]
        scores = cand_sims[order2]

        labels = [self.options[int(i)] for i in final_idx]
        margin = float(scores[0]) - float(scores[1]) if len(scores) >= 2 else 0.0

        return labels, scores.astype(np.float32), {
            "final_top1": labels[0] if labels else None,
            "margin": margin,
        }
