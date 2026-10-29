"""Walk-forward in-context study of Nori-1.5 on the 300-stock cross-section.

Each calendar month is one prediction block. The context is a sample of labeled
(date, stock) rows whose forward target ends before the block starts. Predictions
are made for every stock-day in the block. Results are written per block so a
run can be resumed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# The GPU box belongs to someone else, so the BLAS thread pools are pinned to
# a single thread BEFORE numpy is imported: OpenBLAS reads these at import
# time, and setting them afterwards changes nothing.
_THREADS = os.environ.get("NORI_THREADS", "")
if _THREADS:
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_var] = _THREADS

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

FEATURE_SETS = {
    "full": common.FEATURES,
    "price": ["r1", "r5", "r20", "r60", "r120", "vol20", "vol60", "rsi14", "gap20", "gap60", "dd120", "intraday", "volz"],
    "mom": ["r5", "r20", "r60", "vol20", "xs_r5", "xs_r20", "xs_vol20"],
    "xs": ["xs_r5", "xs_r20", "xs_vol20", "vol20", "rsi14"],
}


def month_starts(dates: pd.DatetimeIndex, start: str, end: str):
    d = pd.Series(dates)
    sel = (d >= pd.Timestamp(start)) & (d < pd.Timestamp(end))
    idx = np.where(sel)[0]
    if len(idx) == 0:
        return []
    months = d.iloc[idx].dt.to_period("M")
    out = []
    for period, grp in d.iloc[idx].groupby(months):
        out.append((str(period), int(grp.index[0]), int(grp.index[-1]) + 1))
    return out


def build_context(feats, target, names, ctx_idx, max_ctx, rng, mode="uniform", scheme="rows", n_assets=300):
    if scheme == "spread":
        n_dates = max(2, max_ctx // max(1, n_assets))
        if len(ctx_idx) > n_dates:
            pick_dates = np.unique(np.linspace(0, len(ctx_idx) - 1, n_dates).round().astype(int))
            ctx_idx = np.asarray(ctx_idx)[pick_dates]
    Xs, ys = [], []
    for t in ctx_idx:
        X = np.column_stack([feats[k][t] for k in names])
        y = target[t]
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
        if ok.sum() == 0:
            continue
        Xs.append(X[ok])
        ys.append(y[ok])
    if not Xs:
        return None, None
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    if len(y) > max_ctx:
        if mode == "recent":
            # ctx_idx is ascending; keep the tail by date ordering of the stacked rows
            y = y[-max_ctx:]
            X = X[-max_ctx:]
        else:
            pick = rng.choice(len(y), max_ctx, replace=False)
            X, y = X[pick], y[pick]
    return X, y


def ridge_predict(Xc, yc, Xq):
    from sklearn.linear_model import Ridge
    mu, sd = Xc.mean(0), Xc.std(0) + 1e-9
    r = Ridge(alpha=10.0).fit((Xc - mu) / sd, yc)
    return r.predict((Xq - mu) / sd)


def limit_cpu(threads: int, low_priority: bool):
    """Keep the job off the box's CPU.

    The GPU box belongs to someone else. Torch's default thread pool is sized
    for the whole machine, so a run that only needs the GPU still grabs every
    core and makes the box feel slow for other work. The BLAS pools are
    already pinned at import time; this pins torch and drops the priority.
    """
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(threads)
    if low_priority and os.name == "nt":
        import ctypes
        BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
        k32 = ctypes.windll.kernel32
        k32.SetPriorityClass(k32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS)
    try:
        import torch
        torch.set_num_threads(threads)
    except Exception:
        pass


def cap_gpu_process(gb: float) -> float:
    """Hard ceiling on this process's share of the card, in GiB.

    The library's own ``gpu_budget_absolute_gb`` sizes the K/V cache and the
    element budget sizes one forward, but neither bounds the caching allocator,
    which may hold more device memory than the tensors alive at any moment.
    PyTorch does expose a process-level cap, and it is the only one of the
    three that cannot be exceeded: past the fraction, allocation fails loudly
    instead of quietly taking the card.
    """
    import torch
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(gb / total)
    return total


def vram_peak_mb() -> float:
    """Peak device memory this process asked PyTorch for, in MB.

    The card is shared with its owner, so the number the process reports has
    to be the number that was really taken, not the one the policy promised.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return float("nan")
        return torch.cuda.max_memory_reserved() / 2 ** 20
    except Exception:
        return float("nan")


def memory_policy(args) -> dict:
    """The library's own budget object, spelled in absolute GiB.

    The box has a fixed 8 GiB card that is shared, so a fraction of the card
    is the wrong unit: what matters is a hard ceiling. The host ceiling is
    set too, because the one run that made the box slow was a cache that
    spilled into system RAM rather than a slow GPU.
    """
    pol = {"gpu_budget_frac": args.gpu_budget_frac,
           "allow_subsample": not args.no_subsample}
    if args.gpu_budget_gb is not None:
        pol["gpu_budget_absolute_gb"] = args.gpu_budget_gb
    if args.host_budget_gb is not None:
        pol["host_budget_absolute_gb"] = args.host_budget_gb
    if args.elements_budget is not None:
        # The GPU ceiling above sizes the K/V cache, which this study never
        # builds, so it never binds. The element budget is what decides the
        # number of rows in one forward, and therefore the real footprint.
        pol["elements_budget"] = int(args.elements_budget)
    if args.no_offload:
        pol["offload_to_host"] = False
    if args.no_quantization:
        pol["allow_quantization"] = False
    return pol


def as_matrix(raw, n: int, k: int) -> np.ndarray:
    """The library returns (k, n) or (n, k) for a quantile bank; take whichever."""
    a = np.asarray(raw, dtype=float)
    if a.ndim == 1:
        return a.reshape(n, -1)
    if a.shape[0] == n:
        return a.reshape(n, -1)
    return a.reshape(-1, n).T


def per_date_rank(pred: np.ndarray, t_idx: np.ndarray) -> np.ndarray:
    """Cross-sectional percentile of each prediction within its own date."""
    out = np.empty(len(pred), dtype=float)
    for t in np.unique(t_idx):
        m = t_idx == t
        out[m] = pd.Series(pred[m]).rank(pct=True).to_numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--h", type=int, default=20)
    ap.add_argument("--target", default="excess", choices=["excess", "rank", "raw", "abs"])
    ap.add_argument("--features", default="full", choices=list(FEATURE_SETS))
    ap.add_argument("--lookback", type=int, default=1000)
    ap.add_argument("--max-ctx", type=int, default=20000)
    ap.add_argument("--ctx-mode", default="uniform", choices=["uniform", "recent"])
    ap.add_argument("--ctx-scheme", default="rows", choices=["rows", "spread"])
    ap.add_argument("--query-chunk", type=int, default=1000)
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2026-11-01")
    ap.add_argument("--model", default="nori-1-5")
    ap.add_argument("--model-path", default=None,
                    help="local .pt checkpoint; when set the weights are read from that "
                         "file and --model only selects the inference config")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--panel", default="sp500_compact.npz")
    ap.add_argument("--no-nori", action="store_true")
    ap.add_argument("--out-type", default="median", choices=["median", "mean", "quantiles"],
                    help="which point estimate the model returns")
    ap.add_argument("--quantiles", default="0.1,0.5,0.9",
                    help="quantile levels, used when out-type=quantiles")
    ap.add_argument("--draws", type=int, default=1,
                    help="independent context draws per block; the predictions are averaged "
                         "as a cross-sectional rank bag")
    ap.add_argument("--gpu-budget-frac", type=float, default=0.73,
                    help="share of GPU memory the model may use; the box is shared")
    ap.add_argument("--gpu-budget-gb", type=float, default=None,
                    help="hard VRAM ceiling in GiB, overriding the fraction")
    ap.add_argument("--host-budget-gb", type=float, default=None,
                    help="hard ceiling in GiB on host RAM for an offloaded cache")
    ap.add_argument("--elements-budget", type=int, default=None,
                    help="per-forward element cap; the library derives its query "
                         "chunk from it, so this is what really bounds video memory")
    ap.add_argument("--gpu-process-cap-gb", type=float, default=None,
                    help="hard ceiling on this process's share of the card; past it "
                         "PyTorch refuses to allocate rather than taking the GPU")
    ap.add_argument("--no-offload", action="store_true",
                    help="never move the K/V cache into host RAM")
    ap.add_argument("--no-subsample", action="store_true",
                    help="fail loudly instead of silently dropping context rows")
    ap.add_argument("--no-quantization", action="store_true",
                    help="forbid the int8 cache, so the cache is always full precision")
    ap.add_argument("--threads", type=int, default=1,
                    help="torch/numpy thread count; the box is shared with its owner")
    ap.add_argument("--low-priority", action="store_true",
                    help="run the process at below-normal priority so the box stays responsive")
    args = ap.parse_args()

    outdir = common.ART / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    meta_path = outdir / "meta.json"
    done = {int(p.stem.split("_")[-1]) for p in outdir.glob("preds_*.parquet")}
    meta = vars(args)
    meta["started"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    json.dump(meta, open(meta_path, "w"), indent=1)

    dates, tickers, close, open_, volume = common.load_stocks(args.panel)
    feats = common.build_features(close, open_, volume)
    fwd = common.forward_logret(close, args.h)
    if args.target == "excess":
        target = common.excess_target(fwd)
    elif args.target == "rank":
        target = common.rank_target(fwd)
    elif args.target == "abs":
        target = common.abs_target(fwd)
    else:
        target = fwd
    qlv = [float(x) for x in args.quantiles.split(",")] if args.out_type == "quantiles" else []
    names = FEATURE_SETS[args.features]
    blocks = month_starts(pd.DatetimeIndex(dates), args.start, args.end)
    print(f"blocks={len(blocks)} features={len(names)} model={args.model} target={args.target}", flush=True)

    model = None
    if not args.no_nori:
        from synthefy_nori import NoriRegressor
        limit_cpu(args.threads, args.low_priority)
        if args.gpu_process_cap_gb:
            total = cap_gpu_process(args.gpu_process_cap_gb)
            print(f"gpu process cap {args.gpu_process_cap_gb} GiB of {total:.1f} GiB",
                  flush=True)
        t0 = time.time()
        model = NoriRegressor(model=args.model, model_path=args.model_path, device="cuda",
                              memory_policy=memory_policy(args))
        print(f"model loaded in {time.time()-t0:.1f}s policy={memory_policy(args)}", flush=True)

    for bi, (period, a, b) in enumerate(blocks):
        path = outdir / f"preds_{bi:04d}.parquet"
        if bi in done:
            continue
        t_start = time.time()
        ctx_pool = np.arange(max(0, a - args.lookback), max(0, a - args.h))
        if len(ctx_pool) < 20:
            continue
        rows = []
        for t in range(a, b):
            x = np.column_stack([feats[k][t] for k in names])
            ok = np.isfinite(x).all(axis=1) & np.isfinite(fwd[t])
            for j in np.where(ok)[0]:
                rows.append((t, j))
        if not rows:
            continue
        qu = np.array(rows)
        Xq = np.vstack([np.array([feats[k][t, j] for k in names]) for t, j in qu])
        Xq = np.where(np.isfinite(Xq), Xq, 0.0)

        def predict_all(X):
            """Model prediction for a query block, in the requested output form."""
            if args.out_type == "quantiles":
                return as_matrix(model.predict(X, output_type="quantiles", quantiles=qlv),
                                 len(X), len(qlv))
            if args.out_type == "mean":
                return as_matrix(model.predict(X, output_type="mean"), len(X), 1)
            return as_matrix(model.predict(X, output_type="median"), len(X), 1)

        pred = np.full(len(qu), np.nan)
        qcols = {q: np.full(len(qu), np.nan) for q in qlv}
        draw_preds = []
        draw_q = []
        Xc0 = yc0 = None
        for d in range(args.draws):
            drng = np.random.default_rng(args.seed * 1000 + d)
            Xc, yc = build_context(feats, target, names, ctx_pool, args.max_ctx, drng,
                                   args.ctx_mode, args.ctx_scheme, close.shape[1])
            if Xc is None:
                continue
            if Xc0 is None:
                Xc0, yc0 = Xc, yc
            if model is None:
                continue
            model.fit(Xc, yc)
            if args.draws > 1:
                dp = np.full(len(qu), np.nan)
                dq = {q: np.full(len(qu), np.nan) for q in qlv}
                for s in range(0, len(qu), args.query_chunk):
                    blk = predict_all(Xq[s:s + args.query_chunk])
                    dp[s:s + args.query_chunk] = blk[:, 0]
                    for qi, q in enumerate(qlv):
                        dq[q][s:s + args.query_chunk] = blk[:, qi]
                draw_preds.append(dp)
                draw_q.append(dq)
                continue
            for s in range(0, len(qu), args.query_chunk):
                blk = predict_all(Xq[s:s + args.query_chunk])
                if args.out_type == "quantiles":
                    for qi, q in enumerate(qlv):
                        qcols[q][s:s + args.query_chunk] = blk[:, qi]
                pred[s:s + args.query_chunk] = blk[:, 0]

        if Xc0 is None:
            continue
        Xc, yc = Xc0, yc0
        if args.draws > 1 and draw_preds:
            mat = np.vstack([per_date_rank(p, qu[:, 0]) for p in draw_preds])
            pred = mat.mean(axis=0)

        pred_ridge = ridge_predict(Xc, yc, Xq)
        mom = np.array([feats["r20"][t, j] for t, j in qu])

        df = pd.DataFrame({
            "date": dates[qu[:, 0]],
            "t_idx": qu[:, 0],
            "asset": qu[:, 1],
            "ticker": tickers[qu[:, 1]],
            "y": [target[t, j] for t, j in qu],
            "y_raw": [fwd[t, j] for t, j in qu],
            "pred": pred,
            "pred_ridge": pred_ridge,
            "pred_mom": np.where(np.isfinite(mom), mom, 0.0),
        })
        for q, col in qcols.items():
            df[f"q{str(q).replace('.', '')}"] = col
        for d, dp in enumerate(draw_preds):
            df[f"pred_d{d}"] = dp
        for d, dq in enumerate(draw_q):
            for q, col in dq.items():
                df[f"q{str(q).replace('.', '')}_d{d}"] = col
        df.to_parquet(path, index=False)
        if "memory_report" not in meta:
            try:
                meta["memory_report"] = model.memory_report_
                json.dump(meta, open(meta_path, "w"), indent=1)
            except Exception:
                pass
        print(f"[{bi:03d}] {period} rows={len(df)} ctx={len(yc)} secs={time.time()-t_start:.1f} "
              f"vram_mb={vram_peak_mb():.0f}", flush=True)

    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta["vram_peak_mb"] = round(vram_peak_mb(), 1)
    json.dump(meta, open(meta_path, "w"), indent=1)
    print(f"DONE vram_peak_mb={meta['vram_peak_mb']}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
