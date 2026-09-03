# -*- coding: utf-8 -*-
"""Figure generator: renders docs/img/health_degradation.png and
docs/img/rul_pred_vs_truth.png using ONLY real project data
(evaluate.run_trajectories seed=7, same pipeline as reports/evaluation_report.md).
Usage: python tools/make_figs.py  (from anywhere; paths derived from this file)."""
import os
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT, "ai-prediction"))
sys.path.insert(0, os.path.join(PROJECT, "data-ingestion"))

import predictive_engine  # noqa: E402

# avoid touching runtime state under data/ (gitignored, but keep hands off)
predictive_engine.PredictiveEngine.save_state = lambda self, path=None: None

import evaluate  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(PROJECT, "docs", "img")
os.makedirs(OUT, exist_ok=True)

samples, backend = evaluate.run_trajectories(7, evaluate.DEFAULT_CSV)
cm = evaluate.confusion_matrix(samples)
gw = evaluate.golden_window_stats(samples)
rul = evaluate.rul_stats(samples)
thr = rul["vs_failure_threshold"]
print("backend:", backend)
print("accuracy=%.3f macro_f1=%.3f" % (cm["accuracy"], cm["macro_f1"]))
print("golden_miss=%.1f%% not_estimable=%.1f%%" % (gw["stage_normal_pct"], gw["rul_not_estimable_pct"]))
print("median_rel=%.1f%% mae=%.1fh med_abs=%.1fh coverage=%.1f%% n=%d ci_n=%d"
      % (thr["median_rel_err_pct"], thr["mae_hours"], thr["median_abs_err_hours"],
         thr["ci_coverage_pct"], thr["n"], thr["ci_n"]))

# ---------------------------------------------------------------- fig 1
devices = ["MOTOR-001", "FAN-001", "GEARBOX-001"]
colors = {"MOTOR-001": "#d62728", "FAN-001": "#1f77b4", "GEARBOX-001": "#2ca02c"}
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
for ax, did in zip(axes, devices):
    sub = [s for s in samples if s["device_id"] == did]
    cyc = [s["cycle"] for s in sub]
    truth = [100.0 * s["truth_h"] for s in sub]
    score = [s["health_score"] for s in sub]
    ax.axhspan(60, 80, color="#f5c542", alpha=0.18, label="黄金维护窗口(60~80)")
    ax.axhline(35, color="#7f7f7f", ls="--", lw=1, label="失效阈值 35 分")
    ax.plot(cyc, truth, color="#555555", lw=2.2, label="真值健康度(×100)")
    ax.plot(cyc, score, color=colors[did], lw=1.1, alpha=0.85, label="平台校准健康评分")
    ax.set_ylabel("健康评分")
    ax.set_title(did, fontsize=10, loc="right")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=8, ncol=3)
axes[-1].set_xlabel("模拟循环(1 tick = 10 分钟)")
fig.suptitle("健康度退化曲线：校准健康评分 vs 真值健康度(留出轨迹 seed=7, 三台设备全生命周期, n=%d)"
             % cm["n"], fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97))
p1 = os.path.join(OUT, "health_degradation.png")
fig.savefig(p1, dpi=150)
plt.close(fig)
print("saved:", p1)

# ---------------------------------------------------------------- fig 2
fig, ax = plt.subplots(figsize=(7.6, 6.8))
plotted, zero_out = 0, 0
for did in devices:
    xs, ys = [], []
    for s in samples:
        if s["device_id"] != did or s["rul_hours"] is None or not s["truth_rul_thr"]:
            continue
        if s["rul_hours"] <= 0.0:
            zero_out += 1
            continue
        xs.append(s["truth_rul_thr"])
        ys.append(s["rul_hours"])
        plotted += 1
    ax.scatter(xs, ys, s=7, alpha=0.45, color=colors[did], label=did, edgecolors="none")
lim_lo, lim_hi = 0.5, 3000.0
ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="#333333", ls="--", lw=1.2, label="理想预测 y = x")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(lim_lo, lim_hi)
ax.set_ylim(lim_lo, lim_hi)
ax.set_xlabel("真值 RUL(小时, 至失效阈值口径)")
ax.set_ylabel("平台预测 RUL(小时)")
ax.grid(alpha=0.25, which="both")
ax.legend(loc="upper left", fontsize=9)
txt = ("留出轨迹 seed=7, numpy-fallback 后端\n"
       "可估样本 n=%d(图中绘制 %d 点, 另有 %d 点输出 0.0h 未画)\n"
       "80%%CI 实测覆盖率 %.1f%%(名义 80%%, 残差自相关+凸性外推如实偏低)\n"
       "MAE %.1f h / 中位绝对误差 %.1f h / 近失效窗口(≤72h)中位 %.1f h"
       % (thr["n"], plotted, zero_out, thr["ci_coverage_pct"],
          thr["mae_hours"], thr["median_abs_err_hours"],
          thr["near_failure_median_abs_err_hours"]))
ax.text(0.97, 0.05, txt, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8.5, bbox=dict(boxstyle="round,pad=0.45", fc="#fffbe8", ec="#cccc99"))
ax.set_title("RUL 预测 vs 真值(全生命周期散点, 主口径=真值健康度首次跌破 0.35)")
fig.tight_layout()
p2 = os.path.join(OUT, "rul_pred_vs_truth.png")
fig.savefig(p2, dpi=150)
plt.close(fig)
print("saved:", p2)
print("DONE")
