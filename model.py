import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from sympy.physics.units import momentum
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_and_inspect(csv_path, must_have=["xFIP"]):
    """
    Load CSV and run basic checks:
    1) Verify required columns exist
    2) Print shape and a preview of column names
    3) Print count of unique xFIP groups and per-group size stats
    4) Drop NaN and report remaining NaN (should be False)
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded: {csv_path}")
    print(f"Rows: {len(df)} | Cols: {len(df.columns)}")
    print("Columns (first 30):", df.columns.tolist()[:30])

    # Check required columns
    missing = [c for c in must_have if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Group stats
    num_groups = df["xFIP"].nunique()
    print("Unique xFIP groups (pitcher seasons):", num_groups)

    cnt = df["xFIP"].value_counts()
    print("min / mean / max pitches per season:",
          int(cnt.min()), int(cnt.mean()), int(cnt.max()))

    # Drop NaN and report
    df = df.dropna()
    print("Any NaN overall:", df.isna().any().any())

    # Preview
    print(df.head(3))
    return df


def sample_by_player_label(
    df: pd.DataFrame,
    group_cols=("xFIP"),
    # group_cols=("pitcher_id", "season"),   # or ('xFIP',) if xFIP is used as season key
    label_col="label",
    K=32,
    whitelist=("ball", "strike", "base_hit", "field_out"),
    # whitelist=("single", "double", "triple", "home_run",
    #            "ball", "called_strike", "swinging_strike",
    #            "foul", "field_out"),
    prefer_replace_if_short=True,
    random_state=None,
):
    """
    Per group, sample exactly K rows with label-stratified allocation.
    - group_cols: identifiers of a group (e.g., pitcher-season or xFIP)
    - label_col: label column to stratify on (must be in whitelist)
    - K: samples per group
    - prefer_replace_if_short: if True, allow sampling with replacement to fill shortage
    - random_state: RNG seed for reproducibility
    """
    rng = np.random.default_rng(random_state)
    df = df.copy()

    out_parts = []
    for _, g in df.groupby(list(group_cols), sort=False):
        # Restrict to whitelist labels for stratification
        g_wh = g[g[label_col].isin(whitelist)]
        if len(g_wh) == 0:
            # Fallback: random sampling within group
            idx = rng.choice(len(g), size=min(K, len(g)), replace=(len(g) < K))
            out_parts.append(g.iloc[idx])
            continue

        # Label proportions inside group
        counts = g_wh[label_col].value_counts()
        probs = (counts / counts.sum()).reindex(whitelist).fillna(0.0).values
        probs = probs / probs.sum() if probs.sum() > 0 else np.ones(len(whitelist)) / len(whitelist)

        # Allocate with largest remainder to sum exactly K
        raw_alloc = probs * K
        alloc = np.floor(raw_alloc).astype(int)
        remainder = raw_alloc - alloc
        diff = K - alloc.sum()
        if diff > 0:
            order = np.argsort(-remainder)
            for j in order[:diff]:
                alloc[j] += 1
        elif diff < 0:
            order = np.argsort(remainder)
            for j in order[:(-diff)]:
                if alloc[j] > 0:
                    alloc[j] -= 1

        # Stratified draws
        parts = []
        shortage = 0
        for c, need in zip(whitelist, alloc):
            if need <= 0:
                continue
            pool = g_wh[g_wh[label_col] == c]
            n_avail = len(pool)
            if n_avail == 0:
                if prefer_replace_if_short:
                    idx = rng.choice(len(g_wh), size=need, replace=True)
                    parts.append(g_wh.iloc[idx])
                else:
                    shortage += need
                continue

            take = min(need, n_avail)
            idx = rng.choice(n_avail, size=take, replace=False)
            parts.append(pool.iloc[idx])

            remain = need - take
            if remain > 0:
                if prefer_replace_if_short:
                    idx2 = rng.choice(n_avail, size=remain, replace=True)
                    parts.append(pool.iloc[idx2])
                else:
                    shortage += remain

        sel = pd.concat(parts, ignore_index=True) if parts else g_wh.iloc[[]].copy()

        # If still short (when not replacing), fill from remaining pool
        if not prefer_replace_if_short and shortage > 0:
            rest = g_wh.drop(sel.index, errors="ignore")
            if len(rest) == 0:
                rest = g_wh
            idx = rng.choice(len(rest), size=shortage, replace=(len(rest) < shortage))
            sel = pd.concat([sel, rest.iloc[idx]], ignore_index=True)

        # Ensure exactly K
        if len(sel) > K:
            sel = sel.sample(n=K, random_state=rng.integers(1 << 30))
        elif len(sel) < K:
            extras = K - len(sel)
            pool = g_wh if len(g_wh) > 0 else g
            idx = rng.choice(len(pool), size=extras, replace=(len(pool) < extras))
            sel = pd.concat([sel, pool.iloc[idx]], ignore_index=True)

        assert len(sel) == K
        out_parts.append(sel.sample(frac=1.0, random_state=rng.integers(1 << 30)))

    sampled_df = pd.concat(out_parts, ignore_index=True)
    return sampled_df


def run_inspect_and_sample(csv_path, group_cols=('xFIP',), label_col='label', K=32, random_state=42):
    """
    Load CSV, check columns, build a dynamic whitelist based on present labels,
    and perform per-group sampling of exactly K rows.
    """
    df = load_and_inspect(csv_path, must_have=list(set(['xFIP', label_col] + list(group_cols))))

    # Dynamic whitelist based on available labels
    whitelist_all = ("ball", "strike", "base_hit", "field_out")
    labels_in_file = set(df[label_col].unique())
    whitelist = tuple([lab for lab in whitelist_all if lab in labels_in_file])
    print(f"Dynamic whitelist: {whitelist}")

    if len(whitelist) == 0:
        raise ValueError(f"No valid label found in {csv_path}, cannot sample.")

    sampled = sample_by_player_label(
        df,
        group_cols=group_cols,
        label_col=label_col,
        K=K,
        whitelist=whitelist,
        prefer_replace_if_short=True,
        random_state=random_state
    )

    # Normalize group_cols to list for counting
    if isinstance(group_cols, str):
        gcols = [group_cols]
    else:
        gcols = list(group_cols)

    n_groups = sampled[gcols].drop_duplicates().shape[0]
    print(f"Sampling done: {len(sampled)} rows ({n_groups} groups × K={K})")
    return sampled


def test_sampling():
    """
    Simple sanity check to verify per-group K sampling and distribution.
    """
    csv_path = "/Users/ywang/PycharmProjects/PythonProject1/players_processed_bk/data_merged_processed_128plus.csv"

    sampled_df = run_inspect_and_sample(
        csv_path=csv_path,
        group_cols=('xFIP',),       # group by xFIP
        label_col='label',
        K=32,
        random_state=42
    )

    # Show head and per-group counts
    print(sampled_df.head(10))
    counts_per_group = sampled_df.groupby('xFIP').size()
    print("Per-group sample count distribution:\n", counts_per_group.value_counts())


class SeasonFixedKDataset(Dataset):
    """
    Dataset returning (X[K, F], y) per group. Optionally standardizes features and
    resamples K rows at each __getitem__ for stochastic batches.
    """
    def __init__(self, df, feature_cols, group_col='xFIP', K=32,
                 standardize=True, scaler=None, resample=True, seed=46):
        # Reset index to ensure 0..N-1 positional alignment
        df = df.reset_index(drop=True).copy()
        self.feature_cols = feature_cols
        self.group_col = group_col
        self.label_col = group_col
        self.K = K
        self.resample = resample
        self.rng = np.random.default_rng(seed)

        X_all = df[feature_cols].values.astype('float32')
        if scaler is not None:
            self.scaler = scaler
            X_all = self.scaler.transform(df[feature_cols].values).astype('float32')
        elif standardize:
            self.scaler = StandardScaler().fit(X_all)
            X_all = self.scaler.transform(X_all).astype('float32')
        else:
            self.scaler = None

        self._X_groups, self._y_groups = [], []
        # Build grouped arrays with positional indices
        for gid, idxs in df.groupby(group_col, sort=False).groups.items():
            pos = np.asarray(list(idxs))          # positional indices (0..N-1)
            xg = X_all[pos]                       # (Ng, F)
            yg = float(gid)
            self._X_groups.append(xg)
            self._y_groups.append(yg)

        self.feature_dim = self._X_groups[0].shape[-1]

        # Prepare fixed slices for deterministic retrieval when resample=False
        self._fixed_slices = []
        for xg in self._X_groups:
            if len(xg) >= K:
                self._fixed_slices.append(xg[:K])
            else:
                sel = self.rng.choice(len(xg), size=K, replace=True)
                self._fixed_slices.append(xg[sel])

    def __len__(self):
        return len(self._X_groups)

    def __getitem__(self, idx):
        xg = self._X_groups[idx]
        y  = self._y_groups[idx]
        if self.resample:
            N = len(xg)
            sel = self.rng.choice(N, size=self.K, replace=(N < self.K))
            Xk = xg[sel]
        else:
            Xk = self._fixed_slices[idx]
        return torch.tensor(Xk, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def make_loaders(sampled_df, feature_cols, K=32, batch_size=16, seed=46):
    """
    Split by group into train/val/test (no group leakage), fit scaler on train only,
    and return DataLoaders built on SeasonFixedKDataset.
    """
    group_col = 'xFIP'

    # Shuffle and split unique groups
    gids = sampled_df[group_col].drop_duplicates().values
    rng = np.random.default_rng(seed)
    rng.shuffle(gids)
    n = len(gids); n_tr = int(0.7*n); n_val = int(0.15*n)
    tr_ids = set(gids[:n_tr])
    val_ids = set(gids[n_tr:n_tr+n_val])
    te_ids = set(gids[n_tr+n_val:])

    train_df = sampled_df[sampled_df[group_col].isin(tr_ids)].copy()
    val_df   = sampled_df[sampled_df[group_col].isin(val_ids)].copy()
    test_df  = sampled_df[sampled_df[group_col].isin(te_ids)].copy()

    # Fit scaler on train only to avoid leakage
    scaler = StandardScaler().fit(train_df[feature_cols].values.astype('float32'))

    # Build datasets with external scaler and stochastic resampling
    train_ds = SeasonFixedKDataset(
        train_df, feature_cols, group_col=group_col, K=K,
        standardize=False, scaler=scaler,
        resample=True, seed=seed
    )
    val_ds = SeasonFixedKDataset(
        val_df, feature_cols, group_col=group_col, K=K,
        standardize=False, scaler=scaler,
        resample=True, seed=seed+1
    )
    test_ds = SeasonFixedKDataset(
        test_df, feature_cols, group_col=group_col, K=K,
        standardize=False, scaler=scaler,
        resample=True, seed=seed+2
    )

    # DataLoaders
    tr_loader  = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    te_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # Keep compatibility with original return signature
    return tr_loader, val_loader, te_loader, train_ds


class PitchEncoder(nn.Module):
    """
    MLP encoder with residual connection and LayerNorm.
    Input: (B, K, F) -> Output: (B, K, D)
    """
    def __init__(self, input_dim, hidden=128, mid_hidden=96, out=48, dropout=0.2):
        super().__init__()
        self.act  = nn.GELU()

        # Feedforward blocks
        self.lin1 = nn.Linear(input_dim, hidden)
        self.lin2 = nn.Linear(hidden, mid_hidden)
        self.lin3 = nn.Linear(mid_hidden, out)

        self.drop = nn.Dropout(dropout)
        self.ln   = nn.LayerNorm(out)

        # Residual projection if dimension mismatch
        self.proj = nn.Linear(input_dim, out) if input_dim != out else nn.Identity()

    def forward(self, x):  # x: (B, K, F)
        B, K, F = x.shape
        x_flat = x.view(B*K, F)

        h = self.lin1(x_flat)
        h = self.act(h)
        h = self.drop(h)

        h = self.lin2(h)
        h = self.act(h)
        h = self.drop(h)

        h = self.lin3(h)

        # Residual + LayerNorm
        res = self.proj(x_flat)
        h = self.ln(h + res)

        return h.view(B, K, -1)  # (B, K, out)


class AttnPool(nn.Module):
    """
    Two-layer scorer to compute attention weights over K elements.
    Returns pooled vector and attention weights.
    """
    def __init__(self, d, tau=1.0):
        super().__init__()
        self.tau = tau
        self.scorer = nn.Sequential(
            nn.Linear(d, d//2 if d>=2 else 1),
            nn.GELU(),
            nn.Linear(d//2 if d>=2 else 1, 1)
        )

    def forward(self, H):  # H: (B, K, D)
        s = self.scorer(H).squeeze(-1)          # (B, K)
        w = torch.softmax(s / self.tau, dim=1)  # (B, K)
        z = torch.einsum('bk,bkd->bd', w, H)    # (B, D)
        return z, w


class XfipRegressor(nn.Module):
    """
    Encoder + attention pooling + MLP head for scalar regression.
    """
    def __init__(self, input_dim, d_pitch=64, dropout=0.1):
        super().__init__()
        use_stats = False,  # note: keep original logic unchanged
        self.use_stats = use_stats
        self.enc  = PitchEncoder(input_dim, hidden=128, out=d_pitch, dropout=dropout)
        self.pool = AttnPool(d_pitch, tau=0.9)

        in_dim = d_pitch * 5 if use_stats else d_pitch
        self.head = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):                 # x: (B, K, F)
        H = self.enc(x)                   # (B, K, D)
        z, w = self.pool(H)               # z: (B, D), w: (B, K)

        if self.use_stats:
            meanH = H.mean(dim=1)
            stdH  = H.std(dim=1, unbiased=False)
            maxH  = H.max(dim=1).values
            minH  = H.min(dim=1).values
            rngH  = maxH - minH
            feat = torch.cat([z, meanH, stdH, maxH, rngH], dim=-1)  # (B, 5D)
        else:
            feat = z                                                # (B, D)

        yhat = self.head(feat).squeeze(1)                           # (B,)
        return yhat, z, w


def train_once(sampled_df, feature_cols, K=64, batch_size=16, epochs=100, lr=1e-3, device=None):
    """
    Single training run with early stopping, test evaluation, MC averaging,
    and a linear regression baseline for comparison.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tr_loader, val_loader, te_loader, ds = make_loaders(sampled_df, feature_cols, K, batch_size)

    model = XfipRegressor(input_dim=ds.feature_dim, d_pitch=64).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4,  betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.7, patience=10)

    def run_epoch(loader, train=False):
        """Return (MSE, RMSE). RMSE computed from total SE to avoid batch-size bias."""
        model.train(train)
        total_loss, total_se, total_n = 0.0, 0.0, 0
        with torch.set_grad_enabled(train):
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                yhat, _, _ = model(X)
                loss = F.mse_loss(yhat, y)
                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                bs = X.size(0)
                total_loss += loss.item() * bs
                total_se   += torch.sum((yhat - y) ** 2).item()
                total_n    += bs
        mse  = total_loss / max(total_n, 1)
        rmse = (total_se / max(total_n, 1)) ** 0.5
        return mse, rmse

    best_val = float('inf')
    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    patience = 20
    bad = 0

    for ep in range(1, epochs + 1):
        tr_mse, tr_rmse = run_epoch(tr_loader, train=True)
        val_mse, val_rmse = run_epoch(val_loader, train=False)
        sched.step(val_mse)

        print(f"Epoch {ep:02d} | Train MSE {tr_mse:.4f} RMSE {tr_rmse:.3f} | "
              f"Val MSE {val_mse:.4f} RMSE {val_rmse:.3f}")

        if val_mse < best_val - 1e-6:
            best_val = val_mse
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {ep} (no val improvement for {patience} epochs).")
                break

    # Load best weights and evaluate on test
    model.load_state_dict(best_state)
    model.to(device)

    te_mse, te_rmse = run_epoch(te_loader, train=False)
    print(f"Test MSE {te_mse:.4f} | RMSE {te_rmse:.4f}")

    # Monte Carlo averaging across multiple resamples in the Dataset
    MC = 10
    preds_stack = []
    y_true_ref, y_pred_once = collect_predictions(model, te_loader, device)
    preds_stack.append(y_pred_once)
    for _ in range(MC - 1):
        _, y_pred_i = collect_predictions(model, te_loader, device)
        preds_stack.append(y_pred_i)
    y_pred = np.mean(np.stack(preds_stack, axis=0), axis=0)
    y_true = y_true_ref

    test_df = pd.DataFrame({
        "true_xFIP": y_true,
        "pred_xFIP": y_pred
    })
    test_df["error"] = test_df["pred_xFIP"] - test_df["true_xFIP"]
    test_df["abs_error"] = test_df["error"].abs()

    # R^2 computed manually for clarity
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"R^2: {r2:.4f}")

    # Reporting
    print("\n=== Test per-season report (head) ===")
    print(test_df.head(10))
    print("\nSummary:")
    print(test_df[["abs_error"]].describe().T)
    print(f"Computed RMSE check: {float(np.sqrt(((test_df['error'] ** 2).mean()))):.3f} | "
          f"MAE: {float(test_df['abs_error'].mean()):.3f}")

    print("\nWorst by abs_error:")
    print(test_df.sort_values("abs_error", ascending=False).head(50))
    print(test_df.sort_values("abs_error", ascending=True).head(50))

    plot_pred_vs_true(
        y_true, y_pred, title="Model - Predicted vs Actual xFIP",
        save_png="pred_vs_true.png", save_pdf="pred_vs_true.pdf"
    )

    # Linear regression baseline using group-aggregated features
    lr_model, lr_results = run_linear_regression_from_loaders(tr_loader, te_loader, ds, feature_cols)
    print(lr_results.head())

    plot_pred_vs_true(
        lr_results["true_xFIP"],
        lr_results["pred_xFIP"],
        title="Linear Regression - Predicted vs Actual xFIP",
        save_png="linear_regression_pred_vs_true.png",
        save_pdf="linear_regression_pred_vs_true.pdf"
    )

    return model, ds, test_df


def collect_predictions(model, loader, device):
    """
    Collect predictions and targets across a DataLoader without gradient tracking.
    """
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            yhat, _, _ = model(X)
            y_true.append(y.cpu())
            y_pred.append(yhat.cpu())
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    return y_true, y_pred


def predict_single_pitcher(single_pitcher_csv, feature_cols, K, model, dataset, device):
    """
    Predict xFIP for a single pitcher CSV using the same preprocessing and sampling flow.
    """
    # Sample K rows consistently with training flow
    sampled_single = run_inspect_and_sample(
        csv_path=single_pitcher_csv,
        group_cols=("xFIP",),
        label_col="label",
        K=K,
        random_state=42
    )

    # Apply the same scaler
    X_single = sampled_single[feature_cols].values.astype("float32")
    if hasattr(dataset, "scaler") and dataset.scaler is not None:
        X_single = dataset.scaler.transform(X_single).astype("float32")

    # Shape: (1, K, F)
    X_single = torch.tensor(X_single, dtype=torch.float32, device=device).view(1, K, -1)

    # Inference
    model.eval()
    with torch.no_grad():
        yhat, z, w = model(X_single)

    pred_xfip = float(yhat.item())
    true_xfip = float(sampled_single["xFIP"].iloc[0])

    print(f"\nPredicted xFIP: {pred_xfip:.3f} | Actual xFIP: {true_xfip:.3f}")

    # Optional: inspect attention weights
    attn = w.squeeze(0).cpu().numpy()  # (K,)
    top_idx = attn.argsort()[::-1][:5]
    print("\nTop-5 attention weights:")
    for i in top_idx:
        print(f"  Sample index {i} | Weight {attn[i]:.4f}")

    return pred_xfip, true_xfip, attn


def plot_pred_vs_true(y_true, y_pred, title="Predicted vs Actual xFIP",
                      save_png="pred_vs_true.png", save_pdf=None):
    """
    Scatter plot of predicted vs. actual values with a 45-degree reference line.
    Metrics are displayed in the top-left corner.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Metrics
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Axis range
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = 0.05 * (hi - lo + 1e-9)
    lo, hi = lo - pad, hi + pad

    # Plot
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.5)

    # Reference y=x line (style only)
    xline = np.linspace(lo, hi, 100)
    plt.plot(xline, xline, linestyle="--")

    plt.xlabel("Actual xFIP")
    plt.ylabel("Predicted xFIP")
    plt.title(title)
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.gca().set_aspect('equal', adjustable='box')

    # Text box with metrics
    txt = f"RMSE={rmse:.3f}\nMAE={mae:.3f}\nR²={r2:.4f}"
    plt.text(0.05, 0.95, txt, transform=plt.gca().transAxes, va="top", ha="left")

    plt.tight_layout()
    if save_png:
        plt.savefig(save_png, dpi=200)
    if save_pdf:
        plt.savefig(save_pdf)
    plt.show()


def run_linear_regression_from_loaders(tr_loader, te_loader, ds, feature_cols, target_col="xFIP"):
    """
    Train a linear regression baseline on group-aggregated features (mean by default),
    then evaluate on the test set and return predictions.
    """
    # Aggregate one row per group
    train_df = _to_group_df_from_loader(tr_loader, ds, feature_cols, agg="mean")
    test_df  = _to_group_df_from_loader(te_loader, ds, feature_cols, agg="mean")

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df[target_col].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    print(f"[LinearRegression] RMSE: {rmse:.4f} | MAE: {mae:.4f} | R^2: {r2:.4f}")

    results_df = pd.DataFrame({"true_xFIP": y_test, "pred_xFIP": y_pred})
    results_df["error"] = results_df["pred_xFIP"] - results_df["true_xFIP"]
    results_df["abs_error"] = results_df["error"].abs()

    # Summary table for quick inspection
    summary_df = results_df[["abs_error"]].describe(percentiles=[0.25, 0.5, 0.75]).T
    print("\nSummary:")
    print(summary_df)

    return lr, results_df


def _to_group_df_from_loader(loader, ds, feature_cols, agg="mean"):
    """
    Convert SeasonFixedKDataset into a per-group feature row using aggregation.
    Supports:
      1) torch.utils.data.Subset (use indices)
      2) SeasonFixedKDataset (use all groups)
    """
    from torch.utils.data import Subset

    base_ds = loader.dataset
    if isinstance(base_ds, Subset):
        ds_inner = base_ds.dataset
        group_indices = list(base_ds.indices)
    else:
        ds_inner = base_ds
        group_indices = list(range(len(ds_inner)))

    if not hasattr(ds_inner, "_X_groups") or not hasattr(ds_inner, "_y_groups"):
        raise ValueError("Dataset must expose _X_groups and _y_groups for aggregation.")

    rows = []
    for i in group_indices:
        Xg = ds_inner._X_groups[i]  # (Ng, F) already transformed
        y  = ds_inner._y_groups[i]  # scalar (xFIP)

        if agg == "mean":
            feats = Xg.mean(axis=0)
            colnames = feature_cols
        elif agg == "median":
            feats = np.median(Xg, axis=0)
            colnames = feature_cols
        elif agg == "meanstd":
            feats = np.concatenate([Xg.mean(axis=0), Xg.std(axis=0, ddof=0)], axis=0)
            colnames = [f"{c}_mean" for c in feature_cols] + [f"{c}_std" for c in feature_cols]
        else:
            raise ValueError(f"Unsupported agg='{agg}'")

        rows.append({"xFIP": float(y), **{c: float(v) for c, v in zip(colnames, feats)}})

    df_out = pd.DataFrame(rows)
    return df_out


# Entry point
if __name__ == "__main__":
    # Paths and hyperparameters
    csv_path = "/Users/ywang/PycharmProjects/PythonProject1/players_processed_bk/data_merged_processed_256plus.csv"
    # csv_path = "/Users/ywang/PycharmProjects/PythonProject1/players_processed_bk/new_label_256plus.csv"
    K = 64
    BATCH_SIZE = 4
    EPOCHS = 200
    LR = 1e-5

    # Numeric feature list (do not include xFIP or label)
    feature_cols = [
        "effective_speed","release_pos_x","release_pos_y","release_pos_z","release_extension",
        "pfx_x","pfx_z","vx0","vy0","vz0","ax","ay","az","plate_x",
        "release_spin_rate","spin_axis","api_break_z_with_gravity",
        "api_break_x_arm","api_break_x_batter_in"
        # ,"arm_angle","xops_sigmoid"
    ]
    # Alternative feature set example:
    # feature_cols = [
    #     "effective_speed", "release_pos_x", "release_pos_y", "release_pos_z",
    #      "vx0", "vy0", "vz0", "ax", "ay", "az",
    #     "release_spin_rate", "spin_axis",
    #     "api_break_x_arm",  "xops_sigmoid"
    # ]

    # 1) Load and sample K per group
    sampled_df = run_inspect_and_sample(
        csv_path=csv_path,
        group_cols=("xFIP",),
        label_col="label",
        K=K,
        random_state=42
    )

    # Optional: quick distribution check
    per_group = sampled_df.groupby("xFIP").size().value_counts()
    print("Distribution of per-group sample counts:\n", per_group)

    # 2) Train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    model, dataset, test_report = train_once(
        sampled_df=sampled_df,
        feature_cols=feature_cols,
        K=K,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LR,
        device=device
    )

    # 3) Save weights
    save_path = "player_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to: {save_path}")

    # 4) Load weights (sanity check)
    ModelClass = XfipRegressor
    loaded_model = ModelClass(input_dim=dataset.feature_dim, d_pitch=64)
    loaded_model.load_state_dict(torch.load(save_path, map_location=device))
    loaded_model.to(device)
    loaded_model.eval()
    print("Model weights loaded.")

    # 5) Predict on a single group CSV
    predict_single_pitcher(
        single_pitcher_csv="/Users/ywang/PycharmProjects/PythonProject1/players_processed_bk/test1.csv",
        feature_cols=feature_cols,
        K=K,
        model=loaded_model,
        dataset=dataset,
        device=device
    )