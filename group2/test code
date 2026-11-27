#!/usr/bin/env python3
"""Group 2 - Response Time Analysis

需求套件: pandas, numpy, matplotlib

用法:
  python latency_analysis.py --demo                      # 用合成資料測試流程
  python latency_analysis.py trace.csv --format msr      # MSR Cambridge trace
  python latency_analysis.py run_lat.1.log --format fio  # fio latency log
  python latency_analysis.py data.csv --col-op type --col-latency latency \
         --latency-unit us --col-size bytes --col-time ts --time-unit s

注意: 整個輸入檔會一次載入記憶體。多個 trace 請分次執行並指定不同 --outdir。
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

OPS = ["Read", "Write"]
COLORS = {"Read": "#2a78d6", "Write": "#eb6834"}
STYLES = {"Read": "-", "Write": "--"}          # 顏色之外再用線型區分
METRICS = ["mean_ms", "p50_ms", "p95_ms", "p99_ms"]

TO_MS = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3, "100ns": 1e-4}
SIZE_TO_BYTES = {"b": 1, "kb": 1024, "sector": 512}

SIZE_EDGES = [0, 4, 8, 16, 32, 64, 128, np.inf]               # KiB, 右端包含
SIZE_LABELS = ["<=4K", "4-8K", "8-16K", "16-32K", "32-64K", "64-128K", ">128K"]
LAT_EDGES = [0, 0.1, 1, 10, 100, np.inf]                      # ms, 左端包含
LAT_LABELS = ["<0.1 ms", "0.1-1 ms", "1-10 ms", "10-100 ms", ">=100 ms"]

ALIASES = {
    "op": ["op", "type", "operation", "rw", "rwbs", "io_type", "opcode"],
    "latency": ["latency", "latency_ns", "latency_us", "latency_ms", "latency_s",
                "response_time", "responsetime", "response_time_us",
                "response_time_ms", "lat", "lat_us", "lat_ms"],
    "size": ["size", "size_bytes", "bytes", "length", "request_size", "size_kb"],
    "time": ["timestamp", "time", "ts", "issue_time", "arrival_time",
             "timestamp_ns", "timestamp_us", "timestamp_ms", "timestamp_s"],
}


# ---------------------------------------------------------------- 讀取與清理
def find_column(df, kind, explicit):
    if explicit:
        if explicit not in df.columns:
            sys.exit(f"找不到欄位 '{explicit}'。現有欄位: {list(df.columns)}")
        return explicit
    lower = {str(c).strip().lower(): c for c in df.columns}
    return next((lower[a] for a in ALIASES[kind] if a in lower), None)


def unit_from_name(name):
    suffix = str(name).strip().lower().rsplit("_", 1)[-1]
    return suffix if suffix in ("ns", "us", "ms", "s") else None


def load_msr(path):
    """欄位: Timestamp,Hostname,Disk,Type,Offset,Size,ResponseTime (時間單位 100 ns)"""
    names = ["ts", "host", "disk", "op", "offset", "size", "rt"]
    df = pd.read_csv(path, header=None, names=names)
    ts = pd.to_numeric(df["ts"], errors="coerce")
    if len(df) and pd.isna(ts.iloc[0]):                       # 有標題列
        df, ts = df.iloc[1:], ts.iloc[1:]
        ts = pd.to_numeric(df["ts"], errors="coerce")
    if df[["host", "disk"]].drop_duplicates().shape[0] > 1:
        print("警告: 檔案含多個 host/disk，結果為混合統計。", file=sys.stderr)
    return pd.DataFrame({
        "op": df["op"],
        "latency_ms": pd.to_numeric(df["rt"], errors="coerce") * TO_MS["100ns"],
        "size_bytes": pd.to_numeric(df["size"], errors="coerce"),
        "time_s": (ts - ts.min()) * 1e-7,       # 先相減再換算，避免失去精度
    })


def load_fio(path):
    """fio *_lat.log / *_clat.log: time(ms), latency(ns), direction, block size, ...
    需用未做平均的紀錄 (不要設定 log_avg_msec)。時間為紀錄(完成)時間。
    fio 3.0 以前的 latency 單位是 us，請改用 --latency-unit us。
    """
    df = pd.read_csv(path, header=None, skipinitialspace=True)
    if df.shape[1] < 4:
        sys.exit("fio log 至少需要 4 欄: time, value, direction, block size")
    return df


def load_csv(path, args):
    df = pd.read_csv(path)
    cols = {k: find_column(df, k, getattr(args, f"col_{k}")) for k in ALIASES}
    if cols["op"] is None:
        sys.exit(f"找不到讀寫類型欄位，請用 --col-op。現有欄位: {list(df.columns)}")

    def to_seconds(col):
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s):              # 日期時間字串
            t = pd.to_datetime(s, errors="coerce")
            return (t - t.min()).dt.total_seconds()
        unit = args.time_unit or unit_from_name(col)
        if unit is None:
            sys.exit(f"無法判斷 '{col}' 的時間單位，請加 --time-unit")
        return (s - s.min()) * TO_MS[unit] / 1000

    time_s = to_seconds(cols["time"]) if cols["time"] else np.nan

    if args.col_complete:                     # 沒有 latency 欄: 完成時間 - 發出時間
        if not cols["time"]:
            sys.exit("--col-complete 需要搭配 --col-time")
        if args.col_complete not in df.columns:
            sys.exit(f"找不到欄位 '{args.col_complete}'")
        unit = args.time_unit or unit_from_name(cols["time"])
        if unit is None:
            sys.exit("使用 --col-complete 時請加 --time-unit")
        latency = (df[args.col_complete] - df[cols["time"]]) * TO_MS[unit]
    else:
        col = cols["latency"]
        if col is None:
            sys.exit(f"找不到延遲欄位，請用 --col-latency。現有欄位: {list(df.columns)}")
        unit = args.latency_unit or unit_from_name(col)
        if unit is None:                      # 不猜單位，避免整份結果差 1000 倍
            sys.exit(f"無法判斷 '{col}' 的延遲單位，請加 --latency-unit")
        latency = pd.to_numeric(df[col], errors="coerce") * TO_MS[unit]

    size = np.nan
    if cols["size"]:
        unit = args.size_unit or (
            "kb" if str(cols["size"]).lower().endswith("_kb") else "b")
        size = pd.to_numeric(df[cols["size"]], errors="coerce") * SIZE_TO_BYTES[unit]
    print("使用欄位:", {k: v for k, v in cols.items() if v})
    return pd.DataFrame({"op": df[cols["op"]], "latency_ms": latency,
                         "size_bytes": size, "time_s": time_s})


def make_demo(n=60_000, seed=42):
    """合成資料，只用來驗證程式流程，不可當作實驗結果。"""
    rng = np.random.default_rng(seed)
    seconds = np.arange(300)
    rate = 1 + 0.7 * np.sin(seconds / 20) + (rng.random(300) < 0.05) * 2
    sec = rng.choice(seconds, n, p=rate / rate.sum())
    op = rng.choice(OPS, n, p=[0.65, 0.35])
    size = rng.choice([4, 8, 16, 32, 64, 128, 256], n,
                      p=[.40, .20, .15, .10, .08, .05, .02]) * 1024
    base = np.where(op == "Read", 0.35, 0.60)
    lat = (base * rng.lognormal(0, 0.55, n) + size / 1024 * 0.006) * (0.7 + 0.35 * rate[sec])
    tail = rng.random(n) < 0.012
    lat[tail] *= rng.uniform(5, 30, tail.sum())
    return pd.DataFrame({"op": op, "latency_ms": lat, "size_bytes": size,
                         "time_s": sec + rng.random(n)}).sort_values("time_s")


def load(args):
    if args.demo:
        return make_demo()
    path = Path(args.input)
    if not path.is_file():
        sys.exit(f"找不到檔案: {path}")
    if args.format == "msr":
        return load_msr(path)
    if args.format == "fio":
        raw = load_fio(path)
        return pd.DataFrame({
            "op": raw[2],
            "latency_ms": raw[1] * TO_MS[args.latency_unit or "ns"],
            "size_bytes": raw[3],
            "time_s": (raw[0] - raw[0].min()) / 1000,
        })
    return load_csv(path, args)


def clean(df):
    total = len(df)
    key = df["op"].astype(str).str.strip().str.lower()
    op = pd.Series(None, index=df.index, dtype=object)
    op[key.str.startswith("r") | (key == "0")] = "Read"
    op[key.str.startswith("w") | (key == "1")] = "Write"
    df = df.assign(op=op)
    bad_op = df["op"].isna()
    bad_lat = ~bad_op & ~(np.isfinite(df["latency_ms"]) & (df["latency_ms"] >= 0))
    out = df[~bad_op & ~bad_lat].copy()
    out.loc[out["size_bytes"] <= 0, "size_bytes"] = np.nan
    report = {"input_rows": total, "dropped_non_read_write": int(bad_op.sum()),
              "dropped_invalid_latency": int(bad_lat.sum()), "analyzed_rows": len(out)}
    if out.empty:
        sys.exit(f"清理後沒有可分析的資料: {report}")
    return out, report


# ---------------------------------------------------------------- 分析
def summarize(s):
    p50, p95, p99 = np.percentile(s, [50, 95, 99])
    return pd.Series({"count": len(s), "mean_ms": s.mean(), "std_ms": s.std(),
                      "min_ms": s.min(), "p50_ms": p50, "p95_ms": p95,
                      "p99_ms": p99, "max_ms": s.max()})


def latency_stats(df):
    groups = {"All": df}
    groups.update({o: df[df.op == o] for o in OPS if (df.op == o).any()})
    t = pd.DataFrame({k: summarize(v.latency_ms) for k, v in groups.items()}).T
    t["count"] = t["count"].astype(int)
    return t.rename_axis("operation")


def rw_comparison(stats):
    if not set(OPS) <= set(stats.index):
        return None
    t = stats.loc[OPS, METRICS].T
    t["write_minus_read_ms"] = t["Write"] - t["Read"]
    t["write_over_read"] = t["Write"] / t["Read"].replace(0, np.nan)
    return t.rename_axis("metric")


def distribution(df):
    bucket = pd.cut(df.latency_ms, LAT_EDGES, labels=LAT_LABELS, right=False)
    t = pd.crosstab(bucket, df.op, normalize="columns") * 100
    return t.reindex(LAT_LABELS, fill_value=0).rename_axis("latency_range")


def spearman(a, b):
    pair = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(pair) < 3 or pair.a.nunique() < 2 or pair.b.nunique() < 2:
        return np.nan
    r = pair.rank()
    return r.a.corr(r.b)


def size_analysis(df):
    d = df.dropna(subset=["size_bytes"])
    if d.empty:
        return None, {}
    d = d.assign(size_bin=pd.cut(d.size_bytes / 1024, SIZE_EDGES, labels=SIZE_LABELS))
    t = (d.groupby(["op", "size_bin"], observed=True).latency_ms
         .apply(summarize).unstack()[["count"] + METRICS])
    t["count"] = t["count"].astype(int)
    corr = {o: spearman(g.size_bytes, g.latency_ms) for o, g in d.groupby("op")}
    return t, corr


def intensity_analysis(df, window):
    d = df.dropna(subset=["time_s"])
    if d.empty:
        return None, None, {}
    idx = (d.time_s // window).astype(int)
    g = d.assign(window=idx).groupby("window")
    w = pd.DataFrame({
        "requests": g.size(),
        "iops": g.size() / window,
        "throughput_MBps": g.size_bytes.sum(min_count=1) / window / 1e6,
        "mean_ms": g.latency_ms.mean(),
        "p50_ms": g.latency_ms.median(),
        "p95_ms": g.latency_ms.quantile(0.95),
    })
    # 最後一個時間窗通常不完整，會低估 IOPS，所以排除
    if len(w) > 1 and d.time_s.max() < (w.index[-1] + 0.99) * window:
        w = w.iloc[:-1]
    w.insert(0, "start_s", w.index * window)
    corr = {m: spearman(w.iops, w[m]) for m in ["mean_ms", "p50_ms", "p95_ms"]}
    levels = None
    if len(w) >= 4 and w.iops.nunique() >= 2:
        q = pd.qcut(w.iops, 4, duplicates="drop")
        levels = w.groupby(q, observed=True).agg(
            windows=("iops", "size"), avg_iops=("iops", "mean"),
            median_window_mean_ms=("mean_ms", "median"),
            median_window_p95_ms=("p95_ms", "median"))
        levels.index = [f"Level {i + 1}" for i in range(len(levels))]
        levels.index.name = "iops_level (low to high)"
    return w, levels, corr


# ---------------------------------------------------------------- 圖表
def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e5e5e2", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def plain_log(axis):
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def save(fig, path, demo):
    fig.tight_layout()
    if demo:
        fig.text(0.5, 0.5, "SYNTHETIC DATA", ha="center", va="center", rotation=25,
                 fontsize=40, color="#777777", alpha=0.18)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_all(df, stats, size_t, windows, window, outdir, demo):
    ops = [o for o in OPS if o in stats.index]
    # 圖 1: Mean / P50 / P95 / P99
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x, width = np.arange(len(METRICS)), 0.38
    for i, o in enumerate(ops):
        offset = (i - (len(ops) - 1) / 2) * width
        bars = ax.bar(x + offset, stats.loc[o, METRICS].astype(float), width * 0.94,
                      color=COLORS[o], label=o)
        ax.bar_label(bars, fmt="%.3g", fontsize=8, padding=2)
    ax.set_xticks(x, ["Mean", "P50", "P95", "P99"])
    style(ax, "Read vs Write latency", "", "Latency (ms)")
    ax.grid(False, axis="x")
    ax.legend(frameon=False)
    save(fig, outdir / "fig1_rw_comparison.png", demo)

    # 圖 2: 直方圖與 CDF (對數 x 軸)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    pos = df.latency_ms[df.latency_ms > 0]
    if len(pos):
        lo, hi = pos.min(), pos.max()
        bins = np.geomspace(lo, hi if hi > lo else lo * 1.1, 70)
        for o in ops:
            v = np.sort(df.latency_ms[(df.op == o) & (df.latency_ms > 0)])
            if not len(v):
                continue
            axes[0].hist(v, bins=bins, weights=np.full(len(v), 100 / len(v)),
                         histtype="step", linewidth=1.8, color=COLORS[o],
                         linestyle=STYLES[o], label=o)
            axes[1].plot(v, np.arange(1, len(v) + 1) / len(v), color=COLORS[o],
                         linestyle=STYLES[o], linewidth=2, label=o)
    style(axes[0], "Distribution", "Latency (ms, log scale)", "Requests per bin (%)")
    style(axes[1], "CDF", "Latency (ms, log scale)", "Cumulative fraction")
    for p in (0.5, 0.95, 0.99):
        axes[1].axhline(p, color="#999999", linewidth=0.7, linestyle=":")
    for a in axes:
        a.set_xscale("log")
        plain_log(a.xaxis)
        if a.get_legend_handles_labels()[0]:
            a.legend(frameon=False, loc="upper left" if a is axes[0] else "lower right")
    save(fig, outdir / "fig2_distribution.png", demo)

    # 圖 3: request size vs latency
    if size_t is not None:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        for a, m, name in zip(axes, ["p50_ms", "p95_ms", "p99_ms"], ["P50", "P95", "P99"]):
            for o in ops:
                if o in size_t.index.get_level_values(0):
                    v = size_t.loc[o, m].reindex(SIZE_LABELS)
                    a.plot(SIZE_LABELS, v, marker="o" if o == "Read" else "s",
                           color=COLORS[o], linestyle=STYLES[o], linewidth=2, label=o)
            style(a, f"{name} by request size", "Request size (bytes)", "Latency (ms)")
            a.set_ylim(bottom=0)
            a.tick_params(axis="x", rotation=45)
            a.legend(frameon=False)
        save(fig, outdir / "fig3_size_vs_latency.png", demo)

    # 圖 4: I/O intensity vs latency
    if windows is not None and len(windows):
        fig = plt.figure(figsize=(12, 6))
        grid = fig.add_gridspec(2, 2, width_ratios=[1.6, 1])
        a1 = fig.add_subplot(grid[0, 0])
        a2 = fig.add_subplot(grid[1, 0], sharex=a1)
        a3 = fig.add_subplot(grid[:, 1])
        # 沒有 request 的時間窗: IOPS 補 0，延遲留白
        full = windows.reindex(range(windows.index.min(), windows.index.max() + 1))
        t = full.index * window
        a1.plot(t, full.iops.fillna(0), color="#52514e", linewidth=1.2)
        a2.plot(t, full.mean_ms, color="#52514e", linewidth=1.2, marker=".", markersize=3)
        a3.scatter(windows.iops, windows.mean_ms, s=22, alpha=0.6, color="#2a78d6",
                   edgecolors="white", linewidths=0.4)
        style(a1, "I/O intensity over time", "", "IOPS")
        style(a2, "Mean latency over time", "Time (s)", "Latency (ms)")
        style(a3, "IOPS vs mean latency (per window)", "IOPS", "Mean latency (ms)")
        save(fig, outdir / "fig4_intensity_vs_latency.png", demo)


# ---------------------------------------------------------------- 報告
def md_table(t):
    t = t.reset_index()
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "-" if np.isnan(v) else f"{v:,.3f}"
        return str(v).replace("|", "\\|")
    rows = ["| " + " | ".join(map(str, t.columns)) + " |",
            "|" + "---|" * len(t.columns)]
    rows += ["| " + " | ".join(fmt(v) for v in r) + " |"
             for r in t.itertuples(index=False, name=None)]
    return "\n".join(rows)


def strength(r):
    if np.isnan(r):
        return "not computable"
    a = abs(r)
    if a < 0.1:
        return f"negligible (rho = {r:.3f})"
    level = "weak" if a < 0.3 else "moderate" if a < 0.5 else "strong"
    return f"{level} {'positive' if r > 0 else 'negative'} (rho = {r:.3f})"


def findings(stats, dist, size_t, size_corr, levels, int_corr, window):
    out = []
    total = stats.loc["All", "count"]
    for o in OPS:
        if o in stats.index:
            s = stats.loc[o]
            note = " P99 is based on fewer than 100 samples; treat with caution." \
                if s["count"] < 100 else ""
            out.append(
                f"{o}: {int(s['count']):,} requests ({s['count'] / total:.1%}); mean "
                f"{s.mean_ms:.3f} ms, P50 {s.p50_ms:.3f} ms, P95 {s.p95_ms:.3f} ms, "
                f"P99 {s.p99_ms:.3f} ms.{note}")
    if set(OPS) <= set(stats.index):
        for m, name in zip(METRICS, ["mean", "P50", "P95", "P99"]):
            r, w = stats.loc["Read", m], stats.loc["Write", m]
            if r > 0 and w > 0:
                slow, fast = ("Write", "read") if w >= r else ("Read", "write")
                out.append(f"{slow} {name} latency is {max(r, w) / min(r, w):.2f}x "
                           f"the {fast} {name} latency.")
    a = stats.loc["All"]
    if a.p50_ms > 0:
        out.append(f"Tail behaviour (all requests): P99/P50 = {a.p99_ms / a.p50_ms:.1f}x, "
                   f"mean/P50 = {a.mean_ms / a.p50_ms:.2f}x.")
    for o in dist.columns:
        top = dist[o].idxmax()
        out.append(f"{o}: the most common latency range is {top} ({dist[o].max():.1f}% "
                   "of requests).")
    for o, r in size_corr.items():
        text = f"{o}: rank correlation between request size and latency is {strength(r)}."
        g = size_t.loc[o]
        g = g[g["count"] >= 100]            # 樣本太少的分組不拿來下結論
        if len(g) >= 2:
            text += (f" Median latency goes from {g.p50_ms.iloc[0]:.3f} ms ({g.index[0]}) "
                     f"to {g.p50_ms.iloc[-1]:.3f} ms ({g.index[-1]}).")
        out.append(text)
    if int_corr:
        out.append(f"Per {window:g}-second window, correlation of IOPS with mean latency is "
                   f"{strength(int_corr['mean_ms'])}; with P95 latency it is "
                   f"{strength(int_corr['p95_ms'])}.")
    if levels is not None and len(levels) >= 2:
        lo, hi = levels.iloc[0], levels.iloc[-1]
        out.append(f"From the lowest to the highest IOPS level, the median window P95 goes "
                   f"from {lo.median_window_p95_ms:.3f} ms to "
                   f"{hi.median_window_p95_ms:.3f} ms.")
    return out


def main():
    p = argparse.ArgumentParser(description="Group 2 response time analysis")
    p.add_argument("input", nargs="?", help="輸入檔案")
    p.add_argument("--demo", action="store_true", help="使用合成資料")
    p.add_argument("--format", choices=["csv", "msr", "fio"], default="csv")
    for k in ALIASES:
        p.add_argument(f"--col-{k}")
    p.add_argument("--col-complete", help="完成時間欄位 (沒有延遲欄時使用)")
    p.add_argument("--latency-unit", choices=list(TO_MS))
    p.add_argument("--time-unit", choices=list(TO_MS))
    p.add_argument("--size-unit", choices=list(SIZE_TO_BYTES))
    p.add_argument("--window", type=float, default=1.0, help="I/O 強度時間窗 (秒)")
    p.add_argument("--outdir", default="results")
    args = p.parse_args()
    if not args.demo and not args.input:
        p.error("請指定輸入檔案，或使用 --demo")
    if args.window <= 0:
        p.error("--window 必須大於 0")

    df, quality = clean(load(args))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    stats = latency_stats(df)
    compare = rw_comparison(stats)
    dist = distribution(df)
    size_t, size_corr = size_analysis(df)
    windows, levels, int_corr = intensity_analysis(df, args.window)

    tables = {"latency_statistics": stats, "rw_comparison": compare,
              "latency_distribution_pct": dist, "size_vs_latency": size_t,
              "intensity_windows": windows, "intensity_levels": levels}
    for name, t in tables.items():
        if t is not None:
            t.to_csv(outdir / f"{name}.csv")

    plot_all(df, stats, size_t, windows, args.window, outdir, args.demo)

    report = ["# Group 2 - Response Time Analysis: Preliminary Results", ""]
    if args.demo:
        report += ["**SYNTHETIC DEMO DATA - not experimental results.**", ""]
    source = "synthetic demo data" if args.demo else Path(args.input).name
    report += [f"Data source: {source}", "",
               "Data quality: " + ", ".join(f"{k} = {v:,}" for k, v in quality.items()), ""]
    sections = [("Latency statistics", stats), ("Read/Write comparison", compare),
                ("Response time distribution (% of requests)", dist),
                ("Request size vs response time", size_t),
                ("I/O intensity vs response time", levels)]
    for title, t in sections:
        report += [f"## {title}", "",
                   md_table(t) if t is not None else "Not available for this input.", ""]
    report += ["## Preliminary findings", ""]
    report += [f"- {f}" for f in
               findings(stats, dist, size_t, size_corr, levels, int_corr, args.window)]
    report += ["", "Note: correlations describe association only; they do not show "
               "that request size or I/O intensity causes the latency difference.", "",
               "## Preliminary figures", ""]
    report += [f"![{f.stem}]({f.name})\n" for f in sorted(outdir.glob("fig*.png"))]
    (outdir / "group2_results.md").write_text("\n".join(report), encoding="utf-8")

    print("資料品質:", quality)
    print(stats.round(3).to_string())
    print(f"\n完成，結果輸出於 {outdir.resolve()}")


if __name__ == "__main__":
    main()
