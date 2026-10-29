"""Shared data, feature and evaluation helpers for the Nori-1.5 trading study."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ART = ROOT / "artifacts"
REPORT = ROOT / "report"

FEATURES = [
    "r1", "r5", "r20", "r60", "r120",
    "vol20", "vol60", "rsi14", "gap20", "gap60",
    "dd120", "intraday", "volz", "mkt_r20", "mkt_vol20",
    "xs_r5", "xs_r20", "xs_vol20",
]


def load_stocks(path: str = "sp500_compact.npz"):
    d = np.load(DATA / path, allow_pickle=True)
    dates = pd.to_datetime(d["dates"])
    tickers = np.array([str(t) for t in d["tickers"]])
    close = np.asarray(d["close"], dtype=float)
    open_ = np.asarray(d["open"], dtype=float) if "open" in d else np.full_like(close, np.nan)
    volume = np.asarray(d["volume"], dtype=float) if "volume" in d else np.full_like(close, np.nan)
    return dates, tickers, close, open_, volume


def load_crypto():
    d = np.load(DATA / "crypto_daily.npz", allow_pickle=True)
    dates = pd.to_datetime(d["dates"])
    syms = np.array([str(s) for s in d["syms"]])
    close = np.asarray(d["close"], dtype=float).T       # (T, A)
    open_ = np.asarray(d["open"], dtype=float).T
    volume = np.asarray(d["volume"], dtype=float).T
    return dates, syms, close, open_, volume


def _pct_rank(x: np.ndarray) -> np.ndarray:
    """Cross-sectional percentile in [0,1], NaNs preserved."""
    out = np.full_like(x, np.nan, dtype=float)
    for j in range(x.shape[1]):
        col = x[:, j]
        ok = np.isfinite(col)
        if ok.sum() > 2:
            r = pd.Series(col[ok]).rank(pct=True).to_numpy()
            out[ok, j] = r
    return out


def rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    diff = np.diff(close, axis=0, prepend=np.nan)
    up = np.where(diff > 0, diff, 0.0)
    dn = np.where(diff < 0, -diff, 0.0)
    au = pd.DataFrame(up).rolling(n, min_periods=n).mean().to_numpy()
    ad = pd.DataFrame(dn).rolling(n, min_periods=n).mean().to_numpy()
    rs = au / np.where(ad == 0, np.nan, ad)
    return (100.0 / (1.0 + rs)) / 100.0 - 0.5


def build_features(close: np.ndarray, open_: np.ndarray, volume: np.ndarray) -> dict[str, np.ndarray]:
    c = pd.DataFrame(close)
    logc = np.log(c)
    r1 = logc.diff()
    lr = r1.to_numpy()

    def roll_ret(n):
        return (logc - logc.shift(n)).to_numpy()

    vol20 = pd.DataFrame(lr).rolling(20, min_periods=15).std().to_numpy()
    vol60 = pd.DataFrame(lr).rolling(60, min_periods=40).std().to_numpy()
    ma20 = c.rolling(20, min_periods=15).mean().to_numpy()
    ma60 = c.rolling(60, min_periods=40).mean().to_numpy()
    max120 = c.rolling(120, min_periods=80).max().to_numpy()
    vmean20 = pd.DataFrame(volume).rolling(20, min_periods=10).mean().to_numpy()

    f = {
        "r1": lr,
        "r5": roll_ret(5),
        "r20": roll_ret(20),
        "r60": roll_ret(60),
        "r120": roll_ret(120),
        "vol20": vol20,
        "vol60": vol60,
        "rsi14": rsi(close),
        "gap20": close / ma20 - 1.0,
        "gap60": close / ma60 - 1.0,
        "dd120": close / max120 - 1.0,
        "intraday": np.log(close / open_),
        "volz": np.log(np.maximum(volume, 1.0) / np.maximum(vmean20, 1.0)),
    }
    mkt = np.nanmean(lr, axis=1)
    f["mkt_r20"] = np.repeat(pd.Series(mkt).rolling(20, min_periods=15).sum().to_numpy()[:, None], close.shape[1], axis=1)
    f["mkt_vol20"] = np.repeat(pd.Series(mkt).rolling(20, min_periods=15).std().to_numpy()[:, None], close.shape[1], axis=1)
    f["xs_r5"] = _pct_rank(f["r5"])
    f["xs_r20"] = _pct_rank(f["r20"])
    f["xs_vol20"] = _pct_rank(f["vol20"])
    for k, v in f.items():
        f[k] = np.clip(v, -5, 5)
    return f


def forward_logret(close: np.ndarray, h: int) -> np.ndarray:
    logc = np.log(pd.DataFrame(close))
    return (logc.shift(-h) - logc).to_numpy()


def excess_target(fwd: np.ndarray) -> np.ndarray:
    return fwd - np.nanmean(fwd, axis=1, keepdims=True)


def rank_target(fwd: np.ndarray) -> np.ndarray:
    return _pct_rank(fwd) - 0.5


def abs_target(fwd: np.ndarray) -> np.ndarray:
    """Size of the move rather than its direction."""
    return np.abs(fwd)


def stack_rows(dates, tickers, feats, target, start, end):
    """Flatten (T, N) arrays into a row table between two date indices."""
    names = list(feats)
    rows = []
    idx = np.arange(start, end)
    for t in idx:
        x = np.column_stack([feats[k][t] for k in names])
        y = target[t]
        ok = np.isfinite(x).all(axis=1) & np.isfinite(y)
        if ok.sum() < 5:
            continue
        dd = np.full(ok.sum(), t)
        rows.append(pd.DataFrame(np.column_stack([dd, np.arange(len(y))[ok], x[ok], y[ok]]),
                                 columns=["t_idx", "asset", *names, "y"]))
    out = pd.concat(rows, ignore_index=True)
    out["date"] = dates[out["t_idx"].to_numpy(dtype=int)]
    out["ticker"] = tickers[out["asset"].to_numpy(dtype=int)]
    return out


def spearman_ic(df: pd.DataFrame, pred: str = "pred", y: str = "y") -> pd.Series:
    return df.groupby("date").apply(lambda g: g[pred].corr(g[y], method="spearman"), include_groups=False)


def block_bootstrap_mean(x: np.ndarray, block: int = 20, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(x)
    nb = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        means[b] = x[idx[:n]].mean()
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def balanced_accuracy(y_sign: np.ndarray, p_sign: np.ndarray) -> float:
    m = np.isfinite(y_sign) & np.isfinite(p_sign)
    y, p = y_sign[m], p_sign[m]
    if len(y) == 0:
        return np.nan
    pos = y > 0
    neg = y < 0
    a = (p[pos] > 0).mean() if pos.any() else np.nan
    b = (p[neg] < 0).mean() if neg.any() else np.nan
    return float(np.nanmean([a, b]))


def decile_spread(df: pd.DataFrame, pred: str = "pred", y: str = "y", q: float = 0.1) -> pd.DataFrame:
    rows = []
    for d, g in df.groupby("date"):
        if len(g) < 20:
            continue
        lo = g[pred].quantile(q)
        hi = g[pred].quantile(1 - q)
        top = g.loc[g[pred] >= hi, y].mean()
        bot = g.loc[g[pred] <= lo, y].mean()
        rows.append({"date": d, "top": top, "bottom": bot, "spread": top - bot, "n": len(g)})
    return pd.DataFrame(rows)


def save_json(obj, name: str):
    ART.mkdir(parents=True, exist_ok=True)
    with open(ART / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=float)
