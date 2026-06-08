#!/usr/bin/env python3
"""
ONE canonical shared timeline for BOTH the UMAP video and the carrier-free audio.

Defines, exactly once:
  - chronological ordering of the 57 checkpoints
  - per-checkpoint TIME positions on a single [0, DUR] timeline (cp_time[i] = the
    moment checkpoint i is fully "on screen" / fully the active audio descriptor)
  - DUR, FPS

Weighting: every checkpoint gets a base weight + extra weight proportional to the
magnitude of structural change |delta PR| (covariance participation ratio) entering it.
This gives the early-pretrain emergence (PR 28 -> ~13 in the first few stage1 ckpts)
more screen/listening time and compresses the long stable post-pretraining plateau.
The SAME cp_time array drives video frame interpolation AND audio descriptor morph,
so sound and picture stay locked to the same checkpoint at every instant.
"""
import re, os, glob
import numpy as np

DUR = 25.0          # seconds (down from ~47-58)
FPS = 30

def order_key(path):
    n = os.path.basename(path)
    s = (0 if 'stage1' in n else 1 if 'stage2' in n else 2 if 'stage3' in n
         else 3 if 'SFT' in n else 4 if 'DPO' in n
         else 6 if 'RL31' in n else 5 if '_RL_' in n else 9)
    m = re.findall(r'step[_-]?(\d+)', n)
    return (s, int(m[-1]) if m else 0)

def stage_of(l):
    return ("pretrain" if "stage" in l else "SFT" if "SFT" in l else "DPO" if "DPO" in l
            else "RL 3.1" if "RL31" in l else "RL 3.0")

def step_of(l):
    m = re.search(r'step_?(\d+)', l)
    return int(m.group(1)) if m else 0

FILES = sorted(glob.glob('/tmp/colall/*.npz'), key=order_key)
NC = len(FILES)
LABELS = [os.path.basename(f) for f in FILES]
STAGES = [stage_of(l) for l in LABELS]
STEPS = [step_of(l) for l in LABELS]

def _participation_ratio():
    pr = []
    for f in FILES:
        V = np.load(f)['V'].astype(np.float64)
        Cov = (V @ V.T) / V.shape[1]
        ev = np.clip(np.linalg.eigvalsh(Cov), 0.0, None)
        p = ev / (ev.sum() + 1e-12)
        pr.append(1.0 / np.sum(p ** 2))
    return np.array(pr)

def checkpoint_times():
    """Return cp_time (NC,) in [0, DUR]: the time at which each checkpoint is the
    active one. cp_time[0]=0, cp_time[-1]=DUR. Used identically by video and audio."""
    pr = _participation_ratio()
    # weight of the INTERVAL leading INTO checkpoint i (i>=1):
    #   base dwell + extra proportional to structural change |dPR| normalized.
    dpr = np.abs(np.diff(pr))                      # (NC-1,)
    dpr_n = dpr / (dpr.max() + 1e-12)
    base = 1.0
    seg_w = base + 2.2 * dpr_n                     # (NC-1,) early big-change segs longer
    cum = np.concatenate([[0.0], np.cumsum(seg_w)])
    cp_time = cum / cum[-1] * DUR
    return cp_time

# Single source of truth, importable by both renderers.
CP_TIME = checkpoint_times()
