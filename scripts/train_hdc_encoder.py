"""One-shot training of a learned HDC text encoder (replaces SimHash).

Replacement SimHash with a small neural network trained on retrieval pairs
(state, correct option). Uses InfoNCE-style loss with the correct option at
the diagonal of each batch. Trains in a single run ("one-shot").

Usage:
    python scripts/train_hdc_encoder.py --n 2000 --epochs 50
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hd_laya.data_gen import gen_dataset


class HdcEncoder(nn.Module):
    """Learned text->HV encoder."""

    def __init__(self, dims: int = 2048, hidden: int = 2048):
        super().__init__()
        self.dims = dims
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dims),
        )

    def forward(self, x):
        return self.net(x)


def text_to_features(texts, dims=2048, ngram=3, seed=42):
    """TF-IDF-ish features: hashed character n-grams into fixed dims."""
    X = np.zeros((len(texts), dims), dtype=np.float32)
    for i, t in enumerate(texts):
        padded = f"  {t.lower().strip()}  "
        for j in range(len(padded) - ngram + 1):
            gram = padded[j:j+ngram]
            idx = hash(gram) % dims
            sign = 1.0 if hash(gram + str(i)) % 2 == 0 else -1.0
            X[i, idx] += sign
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X = X / norms
    return X


def train_hdc_encoder(n_items=2000, epochs=50, lr=1e-3, device="cuda"):
    """Train HdcEncoder on synthetic high-K retrieval data (one-shot)."""
    # Generate training data: (state, correct_option) pairs
    print("Generating training data...")
    items = gen_dataset(n_items, 20, seed=0)

    X_states = []
    X_options = []
    for it in items:
        X_states.append(it["state"])
        X_options.append(it["questions"]["q"]["criteria"][it["expected"]])

    n_pairs = len(X_states)
    print(f"Training on {n_pairs} (state, correct_option) pairs...")
    print(f"Sample: state='{X_states[0][:60]}', option='{X_options[0][:60]}'")

    model = HdcEncoder(dims=2048, hidden=2048).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Extract features
    print("Extracting features...")
    X_state = torch.tensor(text_to_features(X_states, dims=2048), device=device)
    X_opt = torch.tensor(text_to_features(X_options, dims=2048), device=device)

    print(f"Training: {epochs} epochs, {n_pairs} items...")
    batch_size = 64

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_pairs, device=device)
        total_loss = 0.0

        for i in range(0, n_pairs, batch_size):
            idx = perm[i:i+batch_size]
            state_batch = X_state[idx]
            opt_batch = X_opt[idx]

            state_hv = F.normalize(model(state_batch), dim=1)
            opt_hv = F.normalize(model(opt_batch), dim=1)

            # InfoNCE: diagonal = (state[i], correct_option[i]) = positive
            sims = torch.mm(state_hv, opt_hv.T) / 0.5
            labels = torch.arange(len(idx), device=device)

            optimizer.zero_grad()
            loss = F.cross_entropy(sims, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(perm)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}", flush=True)

    print(f"Training done. Final loss={avg_loss:.4f}")

    # Save model
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "hdc_encoder.pt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"model": model.state_dict(), "dims": 2048, "epochs": epochs, "n_items": n_items}, out_path)
    print(f"Saved to {out_path}")
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    train_hdc_encoder(args.n, args.epochs, device=args.device)
