"""hd-laya probe 5 on Kaggle T4x2 - head-only realtime RLCD update benchmark.

Self-contained (script kernels do not bundle subpackages). Premise: encoder
frozen at deployment, so an online RLCD update = head fwd+bwd+AdamW on cached
encoder hidden states. Benchmarks cpu / cuda:0 / cuda:1.
Writes /kaggle/working/probe5_kaggle.json + DONE marker.
NOTE: Kaggle CPU is 4 cores - not deployment-representative; the T4 GPUs are
the fine-tune/training hardware (2xT4 notebook path).
"""
import json
import statistics
import time

import torch
import torch.nn as nn

D, NHEAD, MARKERS, SAMPLES, REPS, MASKED = 1024, 16, 64, 4, 8, -1e4
SEQS, BATCHES = [192, 512, 1024], [1, 8, 32]
# PART2


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
        return self.scorer(m).squeeze(-1).float().masked_fill(~marker_mask, MASKED)
# PART3


def rlcd_loss(logits, target, mask, sigma=0.5):
    k = mask.sum(-1, keepdim=True).float()
    fmask = mask.float()
    eps = torch.randn((SAMPLES,) + logits.shape, device=logits.device) * sigma * fmask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * fmask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, MASKED), -1)
    logq = torch.log(q.clamp_min(1e-12)).clamp_min(-9.21)
    r = (target.unsqueeze(0) * logq).sum(-1) + 0.75 * (
        (target.unsqueeze(0) * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)
    )
    adv = r - r.mean(0, keepdim=True)
    adv = adv / (adv.std() + 1e-6)
    logp = -(((z - logits.unsqueeze(0)) ** 2) * fmask).sum(-1) / (2 * sigma ** 2)
    ce = -(target * torch.log_softmax(logits.masked_fill(~mask, MASKED), -1) * mask).sum(-1).mean()
    return -(adv * logp).mean() + ce
# PART4


def bench(device_str: str) -> dict:
    torch.manual_seed(0)
    dev = torch.device(device_str)
    head = Head().to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-4)
    out = {"device": device_str, "head_params_m": head.n_params / 1e6, "rows": []}
    for L in SEQS:
        B_MAX = BATCHES[-1]
        h = torch.randn(B_MAX, L, D, device=dev)
        pad = torch.ones(B_MAX, L, dtype=torch.bool, device=dev)
        mp = torch.stack([torch.randperm(L, device=dev)[:MARKERS] for _ in range(B_MAX)])
        mm = torch.ones(B_MAX, MARKERS, dtype=torch.bool, device=dev)
        qt = torch.zeros(B_MAX, dtype=torch.long, device=dev)
        tgt = torch.softmax(torch.randn(B_MAX, MARKERS, device=dev), -1)
        for B in BATCHES:
            sl = lambda t: t[:B]
            for _ in range(3):
                logits = head(sl(h), sl(qt), sl(mp), sl(mm), sl(pad))
                rlcd_loss(logits, sl(tgt), sl(mm)).backward()
                opt.step(); opt.zero_grad(set_to_none=True)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            times = []
            for _ in range(REPS):
                t0 = time.perf_counter()
                logits = head(sl(h), sl(qt), sl(mp), sl(mm), sl(pad))
                rlcd_loss(logits, sl(tgt), sl(mm)).backward()
                opt.step(); opt.zero_grad(set_to_none=True)
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                times.append(time.perf_counter() - t0)
            med = statistics.median(times)
            row = {"seq": L, "batch": B, "update_ms": med * 1e3, "updates_per_s": 1.0 / med}
            out["rows"].append(row)
            print(json.dumps(row), flush=True)
    if dev.type == "cuda":
        out["vram_mb"] = torch.cuda.max_memory_allocated(dev) / 1e6
        torch.cuda.reset_peak_memory_stats(dev)
    return out
def main() -> None:
    # v2: GPUs only - the v1 run already measured Kaggle's 2-thread CPU
    # (seq192/b1 343ms, seq1024/b32 72s); repeating it costs ~39 min of runtime.
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
    env = {
        "torch": torch.__version__,
        "cuda_visible": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        "cpu_threads": torch.get_num_threads(),
    }
    print(json.dumps(env), flush=True)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max(1, len(devices))) as ex:
        outs = list(ex.map(bench, devices))
    result = {"probe": "5-headupdate-kaggle", "env": env, "devices": outs}
    with open("/kaggle/working/probe5_kaggle.json", "w") as f:
        json.dump(result, f, indent=1)
    with open("/kaggle/working/DONE", "w") as f:
        f.write("probe5 complete")
    print("WROTE /kaggle/working/probe5_kaggle.json", flush=True)


if __name__ == "__main__":
    main()
