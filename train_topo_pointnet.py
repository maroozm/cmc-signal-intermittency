#!/usr/bin/env python3
"""
train_topo_pointnet.py

Complete pipeline for identifying weak CMC (Critical Monte Carlo) intermittency
signals in heavy-ion collisions using Topological Data Analysis (TDA) + ML.

Key design choices vs. the previous version
--------------------------------------------
  * 2D periodic Delaunay in (eta, phi) via phantom-point replication, replacing
    the 3D cylindrical embedding.  The Euler formula chi = V - E + F is now
    exact (2D simplicial complex has no tetrahedra), so
        beta_1 = beta_0 - V_active + E_active - F_active
    is correct without correction terms.

  * Betti curves normalised by event track multiplicity N, so all curves live
    in [0, 1] and the trivial amplitude-multiplicity correlation is removed
    from the features before any model sees them.

  * Azimuth-randomised control averaged over N_RAND_REALIZATIONS independent
    realisations (arXiv:2509.02339 methodology), reducing statistical noise
    in the beta^rand baseline.

  * Two feature modes (Config.USE_DELTA_CURVES):
        True  -> train on delta curves  beta^Delta = beta/N - beta_rand/N
        False -> train on raw normalised beta/N
    Delta curves decouple topological structure from multiplicity bias.

  * Clean data layout:
        EPOSDataset.features      (n, 2, steps) — 2-channel ML training input
        EPOSDataset.plot_features (n, 6, steps) — all curves kept for plotting

  * Class balance: both classes capped at MAX_EVENTS_PER_CLASS.
    Set to None to use all available events.

  * Reproducibility: Config.SEED seeds torch, numpy, and random.

  * Cosine-annealing LR scheduler for stable training convergence.

  * New diagnostic plots: multiplicity histogram, individual event Betti
    overlays, ROC curve, confusion matrices.

References
----------
  arXiv:2412.06151  TopoPointNet
  arXiv:2509.02339  Trajectum TDA / azimuth-randomisation methodology

Usage
-----
  pip install uproot awkward torch scipy matplotlib xgboost shap scikit-learn
  python train_topo_pointnet.py
"""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import scipy.spatial
import shap
import torch
import torch.nn as nn
import torch.optim as optim
import uproot
import xgboost as xgb
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset


# ==============================================================================
# 1.  Configuration
# ==============================================================================
class Config:
    # ── Physics cuts (must match ep_cmc_injection.cpp) ─────────────────────
    ETA_MAX = 0.5
    PT_MIN = 0.2
    PT_MAX = 3.0  # consistent with injection; paper uses 2.0 — deliberate deviation
    IST_CODE = 8
    # EPOS4 custom PIDs — NOT standard PDG codes.
    # 120=pi+/-, 130=K+/-, 1120=p/pbar, 2330=Xi, 3331=Omega (and antiparticles)
    EPOS_PIDS = [120, 130, 1120, 2330, 3331]

    # ── Centrality cut (matches ep_cmc_injection.cpp line 508) ─────────────
    # bim is the impact parameter in fm.  bim <= 3.5 selects the most central
    # ~5% of minimum-bias Pb-Pb collisions.  Applying this to the MB background
    # file naturally yields the same ~430 central events used for the signal,
    # without needing a separate output file or an artificial event cap.
    BIM_MAX = 3.5

    # ── Topological parameters ──────────────────────────────────────────────
    NUM_FILTRATION_STEPS = 100
    MAX_EPSILON = 0.5  # max filtration distance in 2D (eta, phi) space
    N_RAND_REALIZATIONS = 5  # phi randomisations averaged per event

    # ── Feature mode ───────────────────────────────────────────────────────
    # True  → train on beta^Delta (removes multiplicity bias — recommended)
    # False → train on raw normalised beta_0, beta_1
    USE_DELTA_CURVES = True

    # ── Class balance ───────────────────────────────────────────────────────
    # The BIM_MAX cut on the MB background file already balances the classes
    # naturally (~430 bg events pass, matching the ~430 signal events).
    # MAX_EVENTS_PER_CLASS caps both classes further — useful for quick tests.
    # Set to None for production; the bim cut then determines the final count.
    MAX_EVENTS_PER_CLASS = 50  # ← set to None for production

    # ── ML hyperparameters ──────────────────────────────────────────────────
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 5e-4  # lower than 1e-3 → more stable with cosine schedule
    DROPOUT_RATE = 0.3
    SEED = 42

    # ── File paths ──────────────────────────────────────────────────────────
    BG_FILE = (
        "/Users/solus/montecarlo.studies/PbPbEPOS/"
        "20250127_060353_2067_109G_1000_200_salman.root"
    )
    SIGNAL_FILE = (
        "/Users/solus/montecarlo.studies/PbPbEPOS/"
        "20250127_060353_2067_109G_1000_200_salman.cmc_lambda0p001.root"
    )

    # ── Device ─────────────────────────────────────────────────────────────
    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


# ==============================================================================
# 2.  Topological feature extraction — 2D periodic Delaunay
# ==============================================================================
def compute_betti_curves_2d(eta, phi, num_steps=100, max_eps=0.5):
    """
    Compute normalised Betti-0 and Betti-1 curves for one event.

    The point cloud lives in 2D (eta, phi) space.  The phi coordinate is
    periodic with period 2*pi; this is handled by phantom-point replication:

        Original points:        (eta_i,  phi_i)           indices 0   .. N-1
        Left phantom copy:      (eta_i,  phi_i - 2*pi)    indices N   .. 2N-1
        Right phantom copy:     (eta_i,  phi_i + 2*pi)    indices 2N  .. 3N-1

    scipy.spatial.Delaunay is run on all 3N points.  Each resulting simplex
    (triangle) is mapped back to original vertex indices via modulo N.
    Simplices whose three mapped indices are not all distinct (phantom of the
    same original point) are discarded.  The remaining edges and triangular
    faces form the periodic 2D Delaunay complex on the cylinder.

    Because the complex is 2-dimensional the Euler formula is exact:
        chi   = V - E + F
        beta1 = beta0 - V_active + E_active - F_active   (no tetrahedra)

    Both output curves are divided by the total track count N to make them
    independent of event multiplicity.

    Parameters
    ----------
    eta, phi  : np.ndarray shape (N,)  — phi in [0, 2*pi]
    num_steps : int
    max_eps   : float

    Returns
    -------
    beta0_norm, beta1_norm : np.ndarray shape (num_steps,)
    """
    N = len(eta)
    if N < 3:
        return np.zeros(num_steps), np.zeros(num_steps)

    # ── 1. Phantom-point replication ────────────────────────────────────────
    eta_3x = np.tile(eta, 3)
    phi_3x = np.concatenate([phi, phi - 2 * np.pi, phi + 2 * np.pi])
    pts_3x = np.column_stack([eta_3x, phi_3x])  # shape (3N, 2)

    # ── 2. 2D Delaunay on 3N points ─────────────────────────────────────────
    try:
        tri = scipy.spatial.Delaunay(pts_3x)
    except Exception as exc:
        print(f"    [WARN] Delaunay failed: {exc}")
        return np.zeros(num_steps), np.zeros(num_steps)

    # ── 3. Extract unique edges and faces in original-index space ───────────
    # tri.simplices has shape (M, 3) for 2D — each row is one triangle
    edges_set = set()
    faces_set = set()
    for s in tri.simplices:
        oa, ob, oc = int(s[0]) % N, int(s[1]) % N, int(s[2]) % N
        # discard degenerate simplices (two vertices are phantoms of same point)
        if oa == ob or ob == oc or oa == oc:
            continue
        face = tuple(sorted([oa, ob, oc]))
        faces_set.add(face)
        edges_set.add((min(oa, ob), max(oa, ob)))
        edges_set.add((min(ob, oc), max(ob, oc)))
        edges_set.add((min(oa, oc), max(oa, oc)))

    if not faces_set:
        return np.zeros(num_steps), np.zeros(num_steps)

    edges_arr = np.array(list(edges_set), dtype=np.int32)  # (E, 2)
    faces_arr = np.array(list(faces_set), dtype=np.int32)  # (F, 3)

    # ── 4. Periodic nearest-neighbour distance (DTFE proxy) ─────────────────
    # Query the 3N-point KDTree from each original point.
    # Skip neighbours that are phantom copies of the same original point.
    kd = scipy.spatial.KDTree(pts_3x)
    k = min(6, 3 * N)
    dd, ii = kd.query(pts_3x[:N], k=k)  # (N, k) — query only originals

    nn_dist = np.full(N, np.inf)
    for i in range(N):
        for j in range(1, k):  # j=0 is the point itself (dist=0)
            if ii[i, j] % N != i:
                nn_dist[i] = dd[i, j]
                break
    # Fallback for the (extremely unlikely) case where all k neighbours are
    # phantoms of i itself.
    inf_mask = np.isinf(nn_dist)
    if inf_mask.any():
        nn_dist[inf_mask] = dd[inf_mask, 1]

    # ── 5. Sub-level-set filtration ─────────────────────────────────────────
    eps_vals = np.linspace(0, max_eps, num_steps)
    beta0_curve = np.zeros(num_steps)
    beta1_curve = np.zeros(num_steps)

    for idx, eps in enumerate(eps_vals):
        active = nn_dist <= eps  # boolean (N,)
        n_active = int(active.sum())
        if n_active == 0:
            continue

        # Active edges (both endpoints active)
        ae_mask = active[edges_arr[:, 0]] & active[edges_arr[:, 1]]
        ae = edges_arr[ae_mask]
        n_ae = len(ae)

        # Active faces (all three vertices active)
        af_mask = (
            active[faces_arr[:, 0]] & active[faces_arr[:, 1]] & active[faces_arr[:, 2]]
        )
        n_af = int(af_mask.sum())

        # Connected components — beta_0
        act_idx = np.where(active)[0]  # positions of True in active
        if n_ae > 0:
            # Re-index to [0, n_active) for the sparse matrix
            local = np.searchsorted(act_idx, ae)
            adj = scipy.sparse.csr_matrix(
                (np.ones(n_ae, dtype=np.float32), (local[:, 0], local[:, 1])),
                shape=(n_active, n_active),
            )
        else:
            adj = scipy.sparse.csr_matrix((n_active, n_active))

        n_comp, _ = scipy.sparse.csgraph.connected_components(adj, directed=False)

        beta0 = n_comp
        # 2D Euler characteristic: chi = V - E + F  =>  beta1 = beta0 - chi
        beta1 = max(0, beta0 - n_active + n_ae - n_af)

        beta0_curve[idx] = beta0
        beta1_curve[idx] = beta1

    # Normalise by track multiplicity N
    return beta0_curve / N, beta1_curve / N


# ==============================================================================
# 3.  ROOT dataset parser
# ==============================================================================
class EPOSDataset(Dataset):
    """
    Loads EPOS4 ROOT events, computes 2D periodic Betti curves, and stores:

    self.features       (n, 2, steps) float32 — ML training input
                        channels: (beta0^Delta, beta1^Delta) if USE_DELTA_CURVES
                                  (beta0/N,     beta1/N)     otherwise

    self.plot_features  (n, 6, steps) float32 — retained for diagnostic plots
                        channel layout:
                          0: beta0/N         (original)
                          1: beta1/N         (original)
                          2: beta0_rand/N    (phi-randomised average)
                          3: beta1_rand/N    (phi-randomised average)
                          4: beta0_delta/N   (original minus randomised)
                          5: beta1_delta/N   (original minus randomised)

    self.multiplicities (n,) int32  — post-cut track count per event
    self.labels         (n,) int64  — 0=background, 1=signal
    """

    def __init__(self, bg_file, sig_file, max_events_per_class=None):
        super().__init__()
        self.features = []
        self.plot_features = []
        self.labels = []
        self.multiplicities = []

        self._load(bg_file, label=0, max_ev=max_events_per_class)
        self._load(sig_file, label=1, max_ev=max_events_per_class)

        self.features = np.array(self.features, dtype=np.float32)
        self.plot_features = np.array(self.plot_features, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        self.multiplicities = np.array(self.multiplicities, dtype=np.int32)

    # ------------------------------------------------------------------
    def _load(self, fpath, label, max_ev):
        if not os.path.exists(fpath):
            print(f"[WARN] File not found: {fpath}")
            return

        cls_name = "Background" if label == 0 else "Signal"
        print(f"\nLoading {cls_name} events from  {os.path.basename(fpath)} ...")

        with uproot.open(fpath) as f:
            tree = f["teposevent0"]
            arrays = tree.arrays(
                ["np", "px", "py", "pz", "id", "ist", "bim"],
            )
            n_ev = len(arrays)
            loaded = 0

            for ev in range(n_ev):
                # centrality cut — matches ep_cmc_injection.cpp line 508
                if float(arrays["bim"][ev]) > Config.BIM_MAX:
                    continue
                # ── track kinematics ──────────────────────────────────────
                np_val = int(arrays["np"][ev])
                px = np.asarray(arrays["px"][ev][:np_val], dtype=float)
                py = np.asarray(arrays["py"][ev][:np_val], dtype=float)
                pz = np.asarray(arrays["pz"][ev][:np_val], dtype=float)
                pids = np.asarray(arrays["id"][ev][:np_val], dtype=int)
                ists = np.asarray(arrays["ist"][ev][:np_val], dtype=int)

                pt = np.sqrt(px**2 + py**2)
                pt_safe = np.where(pt > 0, pt, 1e-9)
                eta = np.arcsinh(pz / pt_safe)
                phi = np.arctan2(py, px)
                phi = np.where(phi < 0, phi + 2 * np.pi, phi)

                # ── kinematic cuts ────────────────────────────────────────
                mask = (
                    np.isin(np.abs(pids), Config.EPOS_PIDS)
                    & (ists == Config.IST_CODE)
                    & (np.abs(eta) < Config.ETA_MAX)
                    & (pt > Config.PT_MIN)
                    & (pt < Config.PT_MAX)
                )
                eta_c = eta[mask]
                phi_c = phi[mask]
                N_c = len(eta_c)

                if N_c < 4:
                    continue

                # ── original Betti curves ─────────────────────────────────
                try:
                    b0, b1 = compute_betti_curves_2d(
                        eta_c,
                        phi_c,
                        num_steps=Config.NUM_FILTRATION_STEPS,
                        max_eps=Config.MAX_EPSILON,
                    )
                except Exception as exc:
                    print(f"  [WARN] Event {ev}: Betti computation failed — {exc}")
                    continue

                # ── randomised control (averaged over N_RAND_REALIZATIONS) ─
                b0r = np.zeros(Config.NUM_FILTRATION_STEPS, dtype=float)
                b1r = np.zeros(Config.NUM_FILTRATION_STEPS, dtype=float)
                n_ok = 0
                for _ in range(Config.N_RAND_REALIZATIONS):
                    phi_rand = np.random.uniform(0, 2 * np.pi, size=N_c)
                    try:
                        r0, r1 = compute_betti_curves_2d(
                            eta_c,
                            phi_rand,
                            num_steps=Config.NUM_FILTRATION_STEPS,
                            max_eps=Config.MAX_EPSILON,
                        )
                        b0r += r0
                        b1r += r1
                        n_ok += 1
                    except Exception as exc:
                        print(f"  [WARN] Event {ev} rand: Betti failed — {exc}")

                if n_ok == 0:
                    continue
                b0r /= n_ok
                b1r /= n_ok

                # ── delta curves ──────────────────────────────────────────
                b0d = b0 - b0r
                b1d = b1 - b1r

                # ── training features (2-channel) ─────────────────────────
                if Config.USE_DELTA_CURVES:
                    train_feat = np.stack([b0d, b1d], axis=0)
                else:
                    train_feat = np.stack([b0, b1], axis=0)

                # ── plotting features (6-channel) ─────────────────────────
                plot_feat = np.stack([b0, b1, b0r, b1r, b0d, b1d], axis=0)

                self.features.append(train_feat)
                self.plot_features.append(plot_feat)
                self.labels.append(label)
                self.multiplicities.append(N_c)

                loaded += 1
                if loaded % 50 == 0:
                    print(f"  {loaded}/{n_ev} events processed ...")
                if max_ev is not None and loaded >= max_ev:
                    break

            print(f"  Done — {loaded} {cls_name} events loaded.")

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # features already contains the correct 2-channel training input
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])


# ==============================================================================
# 4.  TopoPointNet (1D CNN)
# ==============================================================================
class TopoPointNet(nn.Module):
    """
    1D CNN classifier over Betti curves, following arXiv:2412.06151.
    Input shape: (batch, 2, NUM_FILTRATION_STEPS).
    """

    def __init__(self, in_channels=2, num_classes=2, dropout_rate=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 128, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(256, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)
        self.drop = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, 2, num_steps)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(2)  # (B, 256)
        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.drop(x)
        return self.fc2(x)  # raw logits (B, num_classes)


# ==============================================================================
# 5.  Training / evaluation helpers
# ==============================================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += out.argmax(1).eq(y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            correct += out.argmax(1).eq(y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


def subset_to_numpy(subset):
    """
    Extract training features (2-channel) and labels from a torch.utils.data.Subset.
    Returns numpy arrays with shapes (n, 2, steps) and (n,).
    """
    idx = subset.indices
    ds = subset.dataset
    return ds.features[idx], ds.labels[idx]


# ==============================================================================
# 6.  Diagnostic plot functions
# ==============================================================================


def plot_multiplicity_distribution(dataset, save_path="multiplicity_distribution.png"):
    """
    Histogram of post-cut track multiplicities for background vs signal.

    A large separation here is a red flag: the classifiers could be learning
    N rather than topological structure.  Normalising Betti curves by N and
    training on delta curves are the mitigations; this plot lets you verify
    how severe the problem is in your dataset.
    """
    bg_m = dataset.multiplicities[dataset.labels == 0]
    sig_m = dataset.multiplicities[dataset.labels == 1]

    all_m = np.concatenate([bg_m, sig_m])
    bins = np.linspace(0, all_m.max() * 1.05, 50)

    plt.figure(figsize=(8, 5))
    plt.hist(
        bg_m,
        bins=bins,
        color="black",
        alpha=0.6,
        density=True,
        label=rf"Background (EPOS4)   $\langle N \rangle={bg_m.mean():.0f}$",
    )
    plt.hist(
        sig_m,
        bins=bins,
        color="red",
        alpha=0.6,
        density=True,
        label=rf"Signal (EPOS4 + CMC) $\langle N \rangle={sig_m.mean():.0f}$",
    )
    plt.xlabel("Track multiplicity $N$ per event (after cuts)", fontsize=12)
    plt.ylabel("Probability density", fontsize=12)
    plt.title("Track Multiplicity Distribution: Background vs Signal", fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Multiplicity distribution  →  {save_path}")


def plot_average_betti_curves(dataset):
    """
    3-panel average Betti curves (original / randomised / difference) for
    beta_0 and beta_1, following arXiv:2509.02339 Fig. 3.

    Uses dataset.plot_features (6-channel, all normalised by N):
      ch 0/1  original beta0/beta1
      ch 2/3  phi-randomised beta0_rand/beta1_rand
      ch 4/5  delta curves beta0_delta/beta1_delta
    """
    print("\nPlotting average Betti curves ...")

    pf = dataset.plot_features  # (n, 6, steps)
    labs = dataset.labels
    bg = pf[labs == 0]
    sig = pf[labs == 1]

    if len(bg) == 0 or len(sig) == 0:
        print("  [WARN] Missing one class — skipping average Betti curve plots.")
        return

    eps = np.linspace(0, Config.MAX_EPSILON, Config.NUM_FILTRATION_STEPS)

    def _panel(ax, data_bg, data_sig, ylabel, title, hline=False):
        mb, sb = data_bg.mean(0), data_bg.std(0)
        ms, ss = data_sig.mean(0), data_sig.std(0)
        ax.plot(eps, mb, color="black", lw=2, label="Background (EPOS4)")
        ax.fill_between(eps, mb - sb, mb + sb, color="black", alpha=0.15)
        ax.plot(eps, ms, color="red", lw=2, label="Signal (EPOS4 + CMC)")
        ax.fill_between(eps, ms - ss, ms + ss, color="red", alpha=0.15)
        if hline:
            ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, linestyle=":", alpha=0.5)

    for betti_idx, lbl in [(0, "0"), (1, "1")]:
        ch_orig = betti_idx  # channels in plot_features
        ch_rand = betti_idx + 2
        ch_diff = betti_idx + 4

        fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

        _panel(
            axes[0],
            bg[:, ch_orig],
            sig[:, ch_orig],
            rf"$\beta_{lbl}(\epsilon)/N$",
            rf"Original $\beta_{lbl}$ (normalised by $N$)",
        )

        _panel(
            axes[1],
            bg[:, ch_rand],
            sig[:, ch_rand],
            rf"$\beta_{lbl}^{{\mathrm{{rand}}}}(\epsilon)/N$",
            rf"Azimuth-randomised $\beta_{lbl}$",
        )

        _panel(
            axes[2],
            bg[:, ch_diff],
            sig[:, ch_diff],
            rf"$\beta_{lbl}^\Delta(\epsilon)/N$",
            rf"Difference $\beta_{lbl}^\Delta = \beta_{lbl} - \beta_{lbl}^{{\mathrm{{rand}}}}$",
            hline=True,
        )

        axes[2].set_xlabel(r"Filtration level $\epsilon$", fontsize=11)
        plt.suptitle(
            rf"Average $\beta_{lbl}$ Curves  (2D periodic Delaunay, $|\eta|<0.5$)",
            fontsize=13,
            y=1.01,
        )
        plt.tight_layout()

        fname = f"betti_curves_comparison_beta{lbl}.png"
        plt.savefig(fname, bbox_inches="tight")
        plt.close()
        print(f"  Betti-{lbl} comparison            →  {fname}")


def plot_individual_betti_curves(
    dataset, n_show=20, save_path="betti_individual_curves.png"
):
    """
    Overlay of up to n_show individual normalised Betti curves per class to
    visualise event-by-event spread.  2x2 panel: (beta_0 / beta_1) x (bg / sig).
    """
    pf = dataset.plot_features  # (n, 6, steps); ch 0=b0, 1=b1 (original, normalised)
    labs = dataset.labels
    eps = np.linspace(0, Config.MAX_EPSILON, Config.NUM_FILTRATION_STEPS)

    bg_idx = np.where(labs == 0)[0][:n_show]
    sig_idx = np.where(labs == 1)[0][:n_show]

    if len(bg_idx) == 0 or len(sig_idx) == 0:
        print("  [WARN] Not enough events for individual Betti curve overlay.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)

    panel_cfg = [
        (0, 0, bg_idx, "black", r"$\beta_0/N$ — Background (EPOS4)"),
        (0, 1, sig_idx, "red", r"$\beta_0/N$ — Signal (EPOS4 + CMC)"),
        (1, 0, bg_idx, "steelblue", r"$\beta_1/N$ — Background (EPOS4)"),
        (1, 1, sig_idx, "firebrick", r"$\beta_1/N$ — Signal (EPOS4 + CMC)"),
    ]

    for row, col, idxs, color, title in panel_cfg:
        ax = axes[row][col]
        ch = row  # channel 0 = beta0, channel 1 = beta1
        for i, ev in enumerate(idxs):
            ax.plot(eps, pf[ev, ch], color=color, alpha=0.30 if i > 0 else 0.85, lw=0.9)
        # bold mean line
        ax.plot(
            eps,
            pf[idxs, ch].mean(0),
            color=color,
            lw=2.5,
            label=f"Mean (n={len(idxs)})",
        )
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(rf"$\beta_{row}/N$", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.4)

    for col in range(2):
        axes[1][col].set_xlabel(r"Filtration level $\epsilon$", fontsize=10)

    plt.suptitle(
        f"Individual Event Betti Curves  (up to {n_show} per class, normalised by $N$)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Individual Betti curves            →  {save_path}")


def plot_single_event_evolution(save_path="event_filtration_evolution.png"):
    """
    2D periodic Delaunay filtration evolution for one signal event plotted in
    (eta, phi) space at 4 representative epsilon values.
    Matches Fig. 1 of arXiv:2412.06151.
    """
    print("\nGenerating single-event filtration evolution ...")

    # Re-read the first signal event with enough tracks
    points_2d = None
    with uproot.open(Config.SIGNAL_FILE) as f:
        tree = f["teposevent0"]
        arrays = tree.arrays(["np", "px", "py", "pz", "id", "ist"], entry_stop=50)
        for ev in range(len(arrays)):
            np_val = int(arrays["np"][ev])
            px = np.asarray(arrays["px"][ev][:np_val], dtype=float)
            py = np.asarray(arrays["py"][ev][:np_val], dtype=float)
            pz = np.asarray(arrays["pz"][ev][:np_val], dtype=float)
            pids = np.asarray(arrays["id"][ev][:np_val], dtype=int)
            ists = np.asarray(arrays["ist"][ev][:np_val], dtype=int)

            pt = np.sqrt(px**2 + py**2)
            pt_safe = np.where(pt > 0, pt, 1e-9)
            eta = np.arcsinh(pz / pt_safe)
            phi = np.arctan2(py, px)
            phi = np.where(phi < 0, phi + 2 * np.pi, phi)

            mask = (
                np.isin(np.abs(pids), Config.EPOS_PIDS)
                & (ists == Config.IST_CODE)
                & (np.abs(eta) < Config.ETA_MAX)
                & (pt > Config.PT_MIN)
                & (pt < Config.PT_MAX)
            )
            eta_c = eta[mask]
            phi_c = phi[mask]

            if len(eta_c) >= 150:
                points_2d = np.column_stack([eta_c, phi_c])
                break

    if points_2d is None:
        print("  [WARN] No suitable event found — skipping evolution plot.")
        return

    N = len(points_2d)
    eta_ev = points_2d[:, 0]
    phi_ev = points_2d[:, 1]

    # ── 2D periodic Delaunay ─────────────────────────────────────────────
    eta_3x = np.tile(eta_ev, 3)
    phi_3x = np.concatenate([phi_ev, phi_ev - 2 * np.pi, phi_ev + 2 * np.pi])
    pts_3x = np.column_stack([eta_3x, phi_3x])

    tri = scipy.spatial.Delaunay(pts_3x)

    edges_set = set()
    faces_set = set()
    for s in tri.simplices:
        oa, ob, oc = int(s[0]) % N, int(s[1]) % N, int(s[2]) % N
        if oa == ob or ob == oc or oa == oc:
            continue
        face = tuple(sorted([oa, ob, oc]))
        faces_set.add(face)
        edges_set.add((min(oa, ob), max(oa, ob)))
        edges_set.add((min(ob, oc), max(ob, oc)))
        edges_set.add((min(oa, oc), max(oa, oc)))

    edges_arr = np.array(list(edges_set), dtype=np.int32)
    faces_arr = np.array(list(faces_set), dtype=np.int32)

    # ── Periodic nearest-neighbour distance ──────────────────────────────
    kd = scipy.spatial.KDTree(pts_3x)
    k = min(6, 3 * N)
    dd, ii = kd.query(pts_3x[:N], k=k)
    nn_dist = np.full(N, np.inf)
    for i in range(N):
        for j in range(1, k):
            if ii[i, j] % N != i:
                nn_dist[i] = dd[i, j]
                break
    inf_m = np.isinf(nn_dist)
    if inf_m.any():
        nn_dist[inf_m] = dd[inf_m, 1]

    # ── Plot ─────────────────────────────────────────────────────────────
    epsilons = [0.02, 0.08, 0.18, 0.35]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), sharey=True)

    for pidx, eps in enumerate(epsilons):
        ax = axes[pidx]
        active = nn_dist <= eps
        act_i = np.where(active)[0]
        n_act = len(act_i)

        ae_mask = active[edges_arr[:, 0]] & active[edges_arr[:, 1]]
        ae = edges_arr[ae_mask]
        n_ae = len(ae)

        af_mask = (
            active[faces_arr[:, 0]] & active[faces_arr[:, 1]] & active[faces_arr[:, 2]]
        )
        af = faces_arr[af_mask]
        n_af = int(af_mask.sum())

        # Betti numbers for title annotation
        if n_act == 0:
            b0 = b1 = 0
        else:
            if n_ae > 0:
                loc = np.searchsorted(act_i, ae)
                adj = scipy.sparse.csr_matrix(
                    (np.ones(n_ae, dtype=np.float32), (loc[:, 0], loc[:, 1])),
                    shape=(n_act, n_act),
                )
            else:
                adj = scipy.sparse.csr_matrix((n_act, n_act))
            nc, _ = scipy.sparse.csgraph.connected_components(adj, directed=False)
            b0 = nc
            b1 = max(0, b0 - n_act + n_ae - n_af)

        # Inactive tracks (light grey)
        ax.scatter(eta_ev, phi_ev, color="lightgrey", s=10, zorder=1, rasterized=True)

        # Active tracks (black)
        if n_act > 0:
            ax.scatter(eta_ev[active], phi_ev[active], color="black", s=18, zorder=3)

        # Active edges — skip those that wrap across phi boundary visually
        for e in ae:
            pA, pB = points_2d[e[0]], points_2d[e[1]]
            if abs(pA[1] - pB[1]) < np.pi:
                ax.plot(
                    [pA[0], pB[0]], [pA[1], pB[1]], "k-", lw=0.9, alpha=0.75, zorder=2
                )

        # Active faces (shaded triangles) — skip wrapped faces
        patches = []
        for face in af:
            pA = points_2d[face[0]]
            pB = points_2d[face[1]]
            pC = points_2d[face[2]]
            if (
                abs(pA[1] - pB[1]) < np.pi
                and abs(pB[1] - pC[1]) < np.pi
                and abs(pC[1] - pA[1]) < np.pi
            ):
                patches.append(Polygon(np.stack([pA, pB, pC]), closed=True))
        if patches:
            ax.add_collection(
                PatchCollection(patches, color="lightblue", alpha=0.5, edgecolor="none")
            )

        ax.set_xlabel(r"$\eta$", fontsize=12)
        ax.set_title(
            rf"$\epsilon = {eps:.2f}$"
            "\n"
            rf"$\beta_0 = {b0},\ \beta_1 = {b1}$"
            f"\n({n_act}/{N} active tracks)",
            fontsize=11,
            pad=8,
        )
        ax.set_xlim(-Config.ETA_MAX - 0.05, Config.ETA_MAX + 0.05)
        ax.set_ylim(-0.2, 2 * np.pi + 0.2)
        ax.grid(True, linestyle=":", alpha=0.4)

    axes[0].set_ylabel(r"$\phi$ (rad)", fontsize=12)
    plt.suptitle(
        r"Single-Event Simplicial Complex Evolution — "
        r"2D Periodic Delaunay in $(\eta,\phi)$ (signal event)",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Filtration evolution               →  {save_path}")


def plot_roc_curve(net_targets, net_probs, xgb_probs, save_path="roc_curve.png"):
    """ROC curves for TopoPointNet and XGBoost on the same axes."""
    fpr_n, tpr_n, _ = roc_curve(net_targets, net_probs)
    fpr_x, tpr_x, _ = roc_curve(net_targets, xgb_probs)
    auc_n = roc_auc_score(net_targets, net_probs)
    auc_x = roc_auc_score(net_targets, xgb_probs)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr_n, tpr_n, "b-", linewidth=2, label=f"TopoPointNet  AUC = {auc_n:.4f}")
    plt.plot(fpr_x, tpr_x, "r--", linewidth=2, label=f"XGBoost       AUC = {auc_x:.4f}")
    plt.plot([0, 1], [0, 1], ":", color="grey", linewidth=1, label="Random classifier")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve — TopoPointNet vs XGBoost", fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  ROC curve                          →  {save_path}")


def plot_confusion_matrices(
    net_targets, net_preds, xgb_preds, save_path="confusion_matrices.png"
):
    """Side-by-side confusion matrices for both classifiers."""
    classes = ["Background", "Signal"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, preds, title in [
        (axes[0], net_preds, "TopoPointNet (1D CNN)"),
        (axes[1], xgb_preds, "XGBoost Baseline"),
    ]:
        cm = confusion_matrix(net_targets, preds)
        thresh = cm.max() / 2.0

        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(classes, fontsize=10)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(classes, fontsize=10)
        ax.set_xlabel("Predicted label", fontsize=10)
        ax.set_ylabel("True label", fontsize=10)
        ax.set_title(title, fontsize=12)

        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                    color="white" if cm[i, j] > thresh else "black",
                )

    plt.suptitle("Confusion Matrices — Test Set", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Confusion matrices                 →  {save_path}")


# ==============================================================================
# 7.  Main execution
# ==============================================================================
def main():
    # ── Reproducibility ─────────────────────────────────────────────────────
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    random.seed(Config.SEED)

    print("=" * 65)
    print("  TopoPointNet — CMC Signal Detection Pipeline")
    print("=" * 65)
    print(f"  Device          : {Config.DEVICE}")
    print(
        f"  Training mode   : {'delta (Δβ) curves' if Config.USE_DELTA_CURVES else 'raw normalised β curves'}"
    )
    print(f"  Events/class    : {Config.MAX_EVENTS_PER_CLASS or 'all available'}")
    print(f"  Rand realisations: {Config.N_RAND_REALIZATIONS}")
    print(f"  Seed            : {Config.SEED}")
    print("=" * 65)

    # ── 1. Load dataset ──────────────────────────────────────────────────────
    dataset = EPOSDataset(
        bg_file=Config.BG_FILE,
        sig_file=Config.SIGNAL_FILE,
        max_events_per_class=Config.MAX_EVENTS_PER_CLASS,
    )

    if len(dataset) == 0:
        print("ERROR: Empty dataset — check file paths and track cuts.")
        return

    n_bg = int((dataset.labels == 0).sum())
    n_sig = int((dataset.labels == 1).sum())
    print(f"\nLoaded {len(dataset)} events total: {n_bg} background, {n_sig} signal")
    if n_bg:
        print(
            f"  BG  avg multiplicity: {dataset.multiplicities[dataset.labels == 0].mean():.0f} tracks/event"
        )
    if n_sig:
        print(
            f"  Sig avg multiplicity: {dataset.multiplicities[dataset.labels == 1].mean():.0f} tracks/event"
        )

    # ── 2. Physics diagnostic plots ──────────────────────────────────────────
    print("\n── Physics diagnostic plots ─────────────────────────────────────")
    plot_multiplicity_distribution(dataset)
    plot_average_betti_curves(dataset)
    plot_individual_betti_curves(dataset)
    plot_single_event_evolution()

    # ── 3. Train / test split ────────────────────────────────────────────────
    train_n = int(0.8 * len(dataset))
    test_n = len(dataset) - train_n
    train_ds, test_ds = torch.utils.data.random_split(
        dataset,
        [train_n, test_n],
        generator=torch.Generator().manual_seed(Config.SEED),
    )
    print(f"\nSplit: {len(train_ds)} train  /  {len(test_ds)} test")

    train_ld = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_ld = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    # ── 4. TopoPointNet (1D CNN) ─────────────────────────────────────────────
    model = TopoPointNet(
        in_channels=2,
        num_classes=2,
        dropout_rate=Config.DROPOUT_RATE,
    ).to(Config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    # Cosine annealing reduces LR smoothly to eta_min over EPOCHS — stabilises
    # training and avoids the sharp oscillations seen with a fixed LR.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-5
    )

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    print(f"\n── Training TopoPointNet ({Config.EPOCHS} epochs) ──────────────────────")
    for ep in range(Config.EPOCHS):
        trl, tra = train_epoch(model, train_ld, criterion, optimizer, Config.DEVICE)
        tel, tea = evaluate(model, test_ld, criterion, Config.DEVICE)
        scheduler.step()

        train_losses.append(trl)
        train_accs.append(tra)
        test_losses.append(tel)
        test_accs.append(tea)

        print(
            f"  Epoch {ep + 1:02d}/{Config.EPOCHS}  "
            f"train loss={trl:.4f} acc={tra * 100:.1f}%  "
            f"test  loss={tel:.4f} acc={tea * 100:.1f}%  "
            f"LR={scheduler.get_last_lr()[0]:.2e}"
        )

    # ── 5. Save model ─────────────────────────────────────────────────────────
    torch.save(model.state_dict(), "topo_pointnet.pth")
    print("Model weights saved  →  topo_pointnet.pth")

    # ── 6. Training history plot ──────────────────────────────────────────────
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(train_accs, label="Train", color="blue")
    axs[0].plot(test_accs, label="Test", color="red", linestyle="--")
    axs[0].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy History")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(train_losses, label="Train", color="blue")
    axs[1].plot(test_losses, label="Test", color="red", linestyle="--")
    axs[1].set(xlabel="Epoch", ylabel="Loss", title="Loss History")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.close()
    print("Training history                   →  training_history.png")

    # ── 7. Collect TopoPointNet predictions on test set ───────────────────────
    model.eval()
    net_probs_list, net_targets_list = [], []
    with torch.no_grad():
        for x, y in test_ld:
            probs = torch.softmax(model(x.to(Config.DEVICE)), dim=1)[:, 1]
            net_probs_list.extend(probs.cpu().numpy())
            net_targets_list.extend(y.numpy())

    net_probs = np.array(net_probs_list)
    net_targets = np.array(net_targets_list)
    net_preds = (net_probs >= 0.5).astype(int)

    # ── 8. XGBoost baseline ───────────────────────────────────────────────────
    X_tr, y_tr = subset_to_numpy(train_ds)
    X_te, y_te = subset_to_numpy(test_ds)
    X_tr_flat = X_tr.reshape(X_tr.shape[0], -1)  # (n_train, 200)
    X_te_flat = X_te.reshape(X_te.shape[0], -1)  # (n_test,  200)

    print("\n── Training XGBoost baseline ────────────────────────────────────")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=Config.SEED,
        eval_metric="logloss",
    )
    xgb_model.fit(X_tr_flat, y_tr)

    xgb_preds = xgb_model.predict(X_te_flat)
    xgb_probs = xgb_model.predict_proba(X_te_flat)[:, 1]

    # ── 9. Performance comparison table ──────────────────────────────────────
    def _metrics(targets, preds, probs):
        return dict(
            acc=accuracy_score(targets, preds),
            auc=roc_auc_score(targets, probs),
            f1=f1_score(targets, preds, zero_division=0),
        )

    mn = _metrics(net_targets, net_preds, net_probs)
    mx = _metrics(y_te, xgb_preds, xgb_probs)

    print("\n" + "=" * 62)
    print("              MODEL PERFORMANCE  (test set)")
    print("=" * 62)
    print(f"  {'Metric':<10}  {'TopoPointNet (CNN)':>22}  {'XGBoost':>12}")
    print("-" * 62)
    print(f"  {'Accuracy':<10}  {mn['acc'] * 100:>21.2f}%  {mx['acc'] * 100:>11.2f}%")
    print(f"  {'ROC-AUC':<10}  {mn['auc']:>22.4f}  {mx['auc']:>12.4f}")
    print(f"  {'F1-Score':<10}  {mn['f1']:>22.4f}  {mx['f1']:>12.4f}")
    print("=" * 62)

    # ── 10. Evaluation plots ─────────────────────────────────────────────────
    print("\n── Evaluation plots ─────────────────────────────────────────────")
    plot_roc_curve(net_targets, net_probs, xgb_probs)
    plot_confusion_matrices(net_targets, net_preds, xgb_preds)

    # ── 11. SHAP interpretability ─────────────────────────────────────────────
    print("\n── SHAP analysis ────────────────────────────────────────────────")
    ch_lbl = "Δβ" if Config.USE_DELTA_CURVES else "β"
    eps_vals = np.linspace(0, Config.MAX_EPSILON, Config.NUM_FILTRATION_STEPS)
    feature_names = [
        f"{ch_lbl}{ch}(step {s})"
        for ch in range(2)
        for s in range(Config.NUM_FILTRATION_STEPS)
    ]

    try:
        explainer = shap.TreeExplainer(xgb_model)
        shap_vals = explainer.shap_values(X_te_flat)

        # Normalise SHAP output across library versions:
        #   shap < 0.40:  list [shap_class0, shap_class1]  (take index 1)
        #   shap >= 0.40: single ndarray or Explanation     (take as-is)
        if isinstance(shap_vals, list):
            sv = shap_vals[1]
        elif hasattr(shap_vals, "values"):
            sv = shap_vals.values
        else:
            sv = shap_vals

        # Flatten to 2D (n_samples, n_features) if needed
        if sv.ndim == 3:
            sv = sv[:, :, 1]

        # SHAP beeswarm summary
        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv, X_te_flat, feature_names=feature_names, show=False)
        plt.title(
            f"SHAP Feature Importance  (XGBoost, {ch_lbl} curves)", fontsize=13, pad=20
        )
        plt.tight_layout()
        plt.savefig("shap_summary.png")
        plt.close()
        print("  SHAP beeswarm                      →  shap_summary.png")

        # Physics-aligned SHAP importance curve vs epsilon
        mean_abs = np.mean(np.abs(sv), axis=0).reshape(2, Config.NUM_FILTRATION_STEPS)

        plt.figure(figsize=(10, 5))
        plt.plot(
            eps_vals,
            mean_abs[0],
            "b-",
            linewidth=2.5,
            label=f"|{ch_lbl}₀| importance (connected components)",
        )
        plt.plot(
            eps_vals,
            mean_abs[1],
            "r--",
            linewidth=2.5,
            label=f"|{ch_lbl}₁| importance (loops / holes)",
        )
        plt.xlabel(r"Filtration level $\epsilon$", fontsize=12)
        plt.ylabel("Mean |SHAP value|", fontsize=12)
        plt.title(
            f"Topological Scale Importance via SHAP  ({ch_lbl} curves)", fontsize=13
        )
        plt.legend(fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        plt.savefig("shap_betti_importance.png")
        plt.close()
        print("  SHAP Betti importance              →  shap_betti_importance.png")

    except Exception as exc:
        print(f"  [WARN] SHAP computation failed: {exc}")

    # ── 12. TODO: Fq(M²) intermittency analysis on classified events ──────────
    #
    # TODO: After both classifiers are validated, select events predicted as
    #       signal (e.g., net_probs >= 0.7) and run the scaled factorial moment
    #       F_q(M^2) analysis on the isolated signal sample.  Check whether the
    #       intermittency index phi_2 is recovered.
    #
    #       Reference implementation: drawFq.C in this workspace.
    #       Physics target: arXiv:2412.06151 Sec. IV (phi_2 recovery).

    print(f"\nAll outputs saved to  {os.getcwd()}")
    print("Done.")


if __name__ == "__main__":
    main()
