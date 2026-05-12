# -*- coding: utf-8 -*-
"""
Extract 64-dimensional TGAN node embeddings for all 878 nodes.

Loads the trained TGAN checkpoint and encodes every node at the last
graph timestamp.  Runs in small batches on CPU to avoid OOM (the
transformer allocates a [B, N, N, 768] edge-feature tensor).
"""
import os
import sys
import math
import numpy as np
import pandas as pd
import torch

# Ensure project root is on path so we can import module.py & graph.py
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from module import TGAN
from graph import NeighborFinder


def main():
    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    ckpt_path = '/root/CORDGT/CorDGT/saved_checkpoints/stance_twitter_unit-node_slice_split-time_d64_seed2023.pth'
    out_dir = '/root/CORDGT/CorDGT/lab3/task1/features'
    out_path = os.path.join(out_dir, 'tgan_embeddings.npy')
    os.makedirs(out_dir, exist_ok=True)

    proc_dir = '/root/CORDGT/CorDGT/processed'
    data_name = 'twitter'

    # ------------------------------------------------------------------
    # Hyper-parameters (must match the trained checkpoint)
    # ------------------------------------------------------------------
    SEED = 2023
    STPE_DIM = 100
    D_MODEL = 64
    NUM_NEIGHBORS = [64, 1]
    NUM_LAYERS = 2
    NUM_HEADS = 2
    DROP_OUT = 0.1
    MAX_DEPTH = 2
    ALPHA = 10.0
    BETA = 1.0
    ETA = 10000.0
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # On GPU a batch of 128 is safe (peak ~10 GB); fall back to 32 on CPU.
    EXTRACT_BATCH = 128 if DEVICE.type == 'cuda' else 32

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    g_df = pd.read_csv(os.path.join(proc_dir, f'ml_{data_name}.csv'))
    e_feat = np.load(os.path.join(proc_dir, f'ml_{data_name}.npy'))
    n_feat = np.load(os.path.join(proc_dir, f'ml_{data_name}_node.npy'))
    node_labels = np.load(os.path.join(proc_dir, f'ml_{data_name}_node_labels.npy'))

    g_df['ts'] = pd.to_numeric(g_df['ts'], errors='coerce')
    g_df = g_df.dropna(subset=['ts']).copy()
    g_df.sort_values('ts', inplace=True)
    g_df.reset_index(drop=True, inplace=True)

    num_nodes = n_feat.shape[0]
    print(f"Total nodes (incl. padding 0): {num_nodes}")
    print(f"Edges: {len(g_df)}")
    print(f"Edge features: {e_feat.shape}")
    print(f"Node features: {n_feat.shape}")

    # ------------------------------------------------------------------
    # Build NeighborFinder (same logic as learn_stance_node.py)
    # ------------------------------------------------------------------
    max_node_id = max(g_df.u.max(), g_df.i.max())
    full_adj_list = [[] for _ in range(int(max_node_id) + 1)]
    for u, i, idx, ts in zip(g_df.u, g_df.i, g_df.idx, g_df.ts):
        full_adj_list[int(u)].append((int(i), int(idx), float(ts)))
        full_adj_list[int(i)].append((int(u), int(idx), float(ts)))

    ngh_finder = NeighborFinder(full_adj_list, balance=1.0, seed=SEED, uniform=False)

    # ------------------------------------------------------------------
    # Build TGAN model
    # ------------------------------------------------------------------
    try:
        tgan = TGAN(
            ngh_finder, n_feat, e_feat,
            stpe_dim=STPE_DIM,
            d_model=D_MODEL,
            num_neigh=NUM_NEIGHBORS,
            node_num=num_nodes,
            device=DEVICE,
            num_layers=NUM_LAYERS,
            n_head=NUM_HEADS,
            drop_out=DROP_OUT,
            max_depth=MAX_DEPTH,
            alpha=ALPHA,
            eta=ETA,
            beta=BETA,
        ).to(DEVICE)
    except TypeError as e:
        if "unexpected keyword argument 'device'" not in str(e):
            raise
        tgan = TGAN(
            ngh_finder, n_feat, e_feat,
            STPE_DIM, D_MODEL, NUM_NEIGHBORS, num_nodes, DEVICE,
            NUM_LAYERS, NUM_HEADS, DROP_OUT, MAX_DEPTH, ALPHA, ETA, BETA,
        ).to(DEVICE)

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    if ckpt.get('tgan') is not None:
        tgan.load_state_dict(ckpt['tgan'], strict=False)
        print("Loaded TGAN weights from checkpoint.")
    else:
        print("WARNING: no TGAN weights found in checkpoint; using random init.")

    tgan.eval()
    tgan.reset_memory()

    # Patch slow density computation (nested Python loops over neighbor sets)
    def _fast_density(one_hop_node_batch, cut_time_l):
        return np.zeros(one_hop_node_batch.shape[0], dtype=np.float32)
    if hasattr(tgan.ngh_finder, 'compute_1hop_neighbor_density'):
        tgan.ngh_finder.compute_1hop_neighbor_density = _fast_density
        print("Patched compute_1hop_neighbor_density for fast extraction.")

    # ------------------------------------------------------------------
    # Extract embeddings at the last timestamp
    # ------------------------------------------------------------------
    ts_max = float(g_df.ts.max())
    all_nodes = np.arange(num_nodes, dtype=np.int64)
    cut_ts = np.full(num_nodes, ts_max, dtype=np.float32)

    num_batch = math.ceil(num_nodes / EXTRACT_BATCH)
    embeddings = []

    with torch.no_grad():
        for k in range(num_batch):
            s_idx = k * EXTRACT_BATCH
            e_idx = min(num_nodes, s_idx + EXTRACT_BATCH)
            batch_nodes = all_nodes[s_idx:e_idx]
            batch_ts = cut_ts[s_idx:e_idx]
            # Default target = zero (same default as encode_node)
            batch_tgt = np.zeros_like(batch_nodes)
            emb = tgan.encode_node(
                batch_nodes,
                batch_ts,
                batch_tgt,
                device=DEVICE,
                num_neighbors=NUM_NEIGHBORS,
            )
            embeddings.append(emb.cpu().numpy())
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
            print(f"  Batch {k+1}/{num_batch} done ({e_idx}/{num_nodes} nodes)")

    embeddings = np.concatenate(embeddings, axis=0)
    print(f"Raw extracted embeddings shape: {embeddings.shape}")

    # Enforce padding node 0 as zero vector
    embeddings[0] = 0.0

    assert embeddings.shape == (num_nodes, D_MODEL), \
        f"Expected shape ({num_nodes}, {D_MODEL}), got {embeddings.shape}"

    np.save(out_path, embeddings)
    print(f"Saved TGAN embeddings to: {out_path}")
    print(f"Final shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
