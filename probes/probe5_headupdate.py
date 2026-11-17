"""Probe 5 - realtime RLCD feasibility: head-only online update cost.

Premise: encoder is FROZEN at deployment (laya's freeze_encoder path), so an
online RLCD update only needs the decision head's fwd+bwd on CACHED encoder
hidden states h - no encoder recompute. This probe times exactly that, on the
real head shapes (d=1024, 2 layers, 4x FFN, nhead=16), with the real RLCD
objective (4 noise samples, proper scoring-rule reward, soft CE).

Run AFTER probes 1-4 (timings must not overlap).
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(exist_ok=True)

D = 1024
NHEAD = 16
SEQS = [192, 512, 1024]
MARKERS = 64
BATCHES = [1, 8, 32]
SAMPLES = 4
REPS = 15
MASKED = -1e4


class Head(nn.Module):
    """DecisionModel minus the encoder (the trainable part)."""

    def __init__(self):
        super().__init__()
        layer = nn.TransformerEncoderLayer(D, NHEAD, 4 * D, 0.1, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.type_emb = nn.Embedding(3, D)
        self.scorer = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D), nn.GELU(), nn.Linear(D, 1))
        self.act_head = nn.Sequential(nn.Linear(D + 4, 256), nn.GELU(), nn.Linear(256, 2))
        self.n_params = sum(p.numel() for p in self.parameters())

    def forward(self, h, qtype, marker_pos, marker_mask, pad_mask):
        h = h + self.type_emb(qtype)[:, None, :]
        h = self.head(h, src_key_padding_mask=~pad_mask.bool())
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, MASKED)
        return logits


def rlcd_loss(logits, target, mask, sigma=0.5):
    k = mask.sum(-1, keepdim=True).float()
    fmask = mask.float()
    eps = torch.randn((SAMPLES,) + logits.shape, device=logits.device) * sigma * fmask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * fmask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, MASKED), -1)
    logq = torch.log(q.clamp_min(1e-12)).clamp_min(-9.21)
    log_score = (target.unsqueeze(0) * logq).sum(-1)
    sph = (target.unsqueeze(0) * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)
    r = log_score + 0.75 * sph
    adv = r - r.mean(0, keepdim=True)
    adv = adv / (adv.std() + 1e-6)
    logp = -(((z - logits.unsqueeze(0)) ** 2) * fmask).sum(-1) / (2 * sigma ** 2)
    logp_ce = torch.log_softmax(logits.masked_fill(~mask, MASKED), -1)
    ce = -(target * logp_ce * mask).sum(-1).mean()
    return -(adv * logp).mean() + ce


def bench(device: str) -> dict:
    torch.manual_seed(0)
    dev = torch.device(device)
    head = Head().to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-4, fused=(device == "cuda"))
    out = {"device": device, "head_params_m": head.n_params / 1e6, "rows": []}
    for L in SEQS:
        h = torch.randn(BATCHES[-1], L, D, device=dev)
        pad = torch.ones(BATCHES[-1], L, dtype=torch.bool, device=dev)
        marker_pos = torch.stack(
            [torch.randperm(L, device=dev)[:MARKERS] for _ in range(BATCHES[-1])]
        )
        marker_mask = torch.ones(BATCHES[-1], MARKERS, dtype=torch.bool, device=dev)
        qtype = torch.zeros(BATCHES[-1], dtype=torch.long, device=dev)
        target = torch.softmax(torch.randn(BATCHES[-1], MARKERS, device=dev), -1)
        mask = marker_mask
        for B in BATCHES:
            sl = lambda t: t[:B]
            for _ in range(3):  # warmup
                logits = head(sl(h), sl(qtype), sl(marker_pos), sl(marker_mask), sl(pad))
                loss = rlcd_loss(logits, sl(target), sl(marker_mask))
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            if device == "cuda":
                torch.cuda.synchronize()
            times = []
            for _ in range(REPS):
                t0 = time.perf_counter()
                logits = head(sl(h), sl(qtype), sl(marker_pos), sl(marker_mask), sl(pad))
                loss = rlcd_loss(logits, sl(target), sl(marker_mask))
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                if device == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            med = statistics.median(times)
            row = {"seq": L, "batch": B, "update_ms": med * 1e3, "updates_per_s": 1.0 / med}
            out["rows"].append(row)
            print(json.dumps(row), flush=True)
    if device == "cuda":
        out["vram_mb"] = torch.cuda.max_memory_allocated() / 1e6
        torch.cuda.reset_peak_memory_stats()
    return out


def main() -> None:
    rows = []
    rows.append(bench("cpu"))
    if torch.cuda.is_available():
        rows.append(bench("cuda"))
    result = {"probe": "5-headupdate", "devices": rows}
    (OUT / "probe5_headupdate.json").write_text(json.dumps(result, indent=1))
    print("WROTE", OUT / "probe5_headupdate.json")


if __name__ == "__main__":
    main()
