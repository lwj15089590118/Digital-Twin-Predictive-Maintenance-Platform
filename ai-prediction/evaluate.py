# -*- coding: utf-8 -*-
"""
================================================================================
 AI 预测引擎定量评估 (Evaluation)
================================================================================
模块职责:
    对预测引擎(健康评分 / 阶段分类 / RUL 置信区间)做留出轨迹定量评估,
    产出混淆矩阵、RUL 误差表、置信区间覆盖率与校准表, 落盘为可复现的
    评估工件(reports/evaluation_report.md + evaluation_metrics.json)。

评估设计:
    - 训练集: data-ingestion/historical_data.csv(模拟器 seed=2026 生成);
    - 评估集: 用不同随机种子(默认 7, 与 --demo 一致)独立仿真三台设备
      的全生命周期轨迹, 与训练集不存在样本重叠(留出法);
    - RUL 真值口径: 主口径为"真值健康度首次跌破失效阈值 35 分对应健康度
      (0.35)的剩余时长", 与 RULPredictor 外推目标的语义一致; 另附"至寿命
      终点(健康度=0)"口径作参考;
    - RUL 量纲契约(v1.2 修正): 真值 RUL 与预测 RUL/置信区间统一为小时——
      真值按 tick 差值 × TICK_HOURS(1 tick = 10 分钟)换算, 唯一换算点在
      truth_rul_hours(); 此前真值滞留 tick 量纲与 rul_hours 直接比较,
      使全部 RUL 定量指标失真(复审报告 07-N-P0-1);
    - 与在线口径的一致性(v1.3 统一, 复审报告 07-N-P1-1): 本评估逐循环喂入
      (1 条记录 = 1 tick), 看板流水线自 v1.3 起同样逐循环把每条记录喂给
      引擎(取消 6:1 节流喂入, 仅展示层与孪生同步取最后一批), 两侧共用
      RULPredictor 的同一换算——同一轨迹上在线 RUL 与本评估逐位一致,
      预测预警(72h 窗口)触发时刻与真值 RUL 对齐; --selftest 含该口径的
      回归断言(逐 tick 口径锁 + 6:1 节流危害用例);
    - 局限声明: 特征与标签同源于自仿真数据(同一健康度决定传感器读数与
      标签), 任何在此数据上的指标都系统性偏高, 不能外推到真实产线精度。

运行示例:
    python evaluate.py                 # 全量评估, 工件写入 reports/
    python evaluate.py --selftest      # RUL 置信区间单元用例(快/慢退化)
================================================================================
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
for sub in (HERE, os.path.join(PROJECT_ROOT, "data-ingestion")):
    if sub not in sys.path:
        sys.path.insert(0, sub)

from predictive_engine import (PredictiveEngine, RULPredictor, HealthScorer,  # noqa: E402
                               load_history, DEFAULT_CSV, TICK_HOURS)
from data_simulator import DataSimulator, DEVICES, SIM_TICK_MINUTES  # noqa: E402

# RUL 真值口径: 失效阈值 35 分在校准评分(≈100×健康度)下对应的真值健康度
TRUTH_FAILURE_HEALTH = 0.35
# 黄金窗口定义: 真值健康度 0.6~0.8(README/模拟器注释所称"预测性维护黄金窗口")
GOLDEN_WINDOW = (0.6, 0.8)
STAGES = ("normal", "warning", "fault")
DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "reports")


def truth_rul_hours(cycle: float, end_cycle: float) -> float:
    """真值 RUL 换算(量纲契约): tick 差值 × TICK_HOURS 统一为小时。

    evaluate.py 内所有真值 RUL(vs 失效阈值 / vs 寿命终点)的唯一 tick→
    小时换算点, 保证与 RULPredictor 输出的 rul_hours / rul_ci_low /
    rul_ci_high(小时)同量纲直接比较; --selftest 含该契约的单元断言。
    v1.0 曾把 tick 差值直接当小时与 rul_hours 比较, 全部 RUL 指标失真
    (复审报告 07-N-P0-1)。
    """
    return max(0.0, float(end_cycle) - float(cycle)) * TICK_HOURS


# ------------------------------------------------------------------------------
# 评估数据采集: 留出轨迹上逐循环推理
# ------------------------------------------------------------------------------
def run_trajectories(seed: int, csv_path: str) -> tuple:
    """在独立种子的全生命周期轨迹上运行引擎, 返回 (逐样本记录列表, 推理后端)。

    每条记录: {device_id, cycle, truth_h, truth_label, health_score,
               stage, rul(estimable/None), rul_hours, ci_low, ci_high}
    量纲契约: truth_rul_thr / truth_rul_eol 均为小时(truth_rul_hours 换算),
    与 rul_hours / ci_low / ci_high 同量纲。
    """
    engine = PredictiveEngine()
    summary = engine.train(csv_path)
    print("[评估] 训练完成: 样本 %d, 后端 %s, sklearn=%s"
          % (summary["total_samples"], engine.backend, summary["sklearn"]))
    sim = DataSimulator(seed=seed)
    # 失效循环(真值健康度首次跌破阈值的循环数), 逐设备在线确定
    fail_cycle = {}
    samples = []
    rounds = max(v["life_cycles"] for v in DEVICES.values()) + 5
    for _ in range(rounds):
        for rec in sim.step_all():
            did = rec["device_id"]
            cycle = rec["cycle"]
            truth_h = rec["health"]
            if did not in fail_cycle and truth_h < TRUTH_FAILURE_HEALTH:
                fail_cycle[did] = cycle
            result = engine.assess(rec)
            rul = result["rul"]
            samples.append({
                "device_id": did,
                "cycle": cycle,
                "truth_h": truth_h,
                "truth_label": rec["label"],
                "health_score": result["health_score"],
                "stage": result["stage"],
                "rul_estimable": rul.get("rul_hours") is not None,
                "rul_hours": rul.get("rul_hours"),
                "ci_low": rul.get("rul_ci_low"),
                "ci_high": rul.get("rul_ci_high"),
                "truth_rul_eol": truth_rul_hours(cycle, DEVICES[did]["life_cycles"]),
            })
    # 真值 RUL(至失效阈值)需在轨迹结束后回填: 失效循环在时序推进中才确定,
    # 早期样本创建时无法预知(流式语义下的经典陷阱); 回填时经 truth_rul_hours
    # 统一换算为小时(复审报告 07-N-P0-1 的量纲修正点)
    for s in samples:
        did = s["device_id"]
        s["truth_rul_thr"] = (truth_rul_hours(s["cycle"], fail_cycle[did])
                              if did in fail_cycle else None)
    return samples, engine.backend


# ------------------------------------------------------------------------------
# 指标计算
# ------------------------------------------------------------------------------
def confusion_matrix(samples: list) -> dict:
    """阶段分类混淆矩阵(行=真值标签, 列=预测阶段)与准确率/宏 F1。"""
    mat = {t: {p: 0 for p in STAGES} for t in STAGES}
    for s in samples:
        if s["truth_label"] in STAGES and s["stage"] in STAGES:
            mat[s["truth_label"]][s["stage"]] += 1
    total = sum(mat[t][p] for t in STAGES for p in STAGES)
    correct = sum(mat[t][t] for t in STAGES)
    per_class = {}
    f1s = []
    for st in STAGES:
        tp = mat[st][st]
        fp = sum(mat[t][st] for t in STAGES) - tp
        fn = sum(mat[st][p] for p in STAGES) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[st] = {"precision": round(precision, 4),
                         "recall": round(recall, 4), "f1": round(f1, 4)}
        f1s.append(f1)
    return {"matrix": mat, "n": total,
            "accuracy": round(correct / total, 4) if total else 0.0,
            "macro_f1": round(sum(f1s) / len(f1s), 4),
            "per_class": per_class}


def golden_window_stats(samples: list) -> dict:
    """黄金窗口(真值健康度 0.6~0.8)专项: 漏检率/RUL 可估性/误差。"""
    lo, hi = GOLDEN_WINDOW
    win = [s for s in samples if lo < s["truth_h"] <= hi]
    n = len(win)
    if not n:
        return {"n": 0}
    stage_normal = sum(1 for s in win if s["stage"] == "normal")
    not_estimable = sum(1 for s in win if not s["rul_estimable"])
    zero_rul = sum(1 for s in win if s["rul_hours"] == 0.0)
    errs = sorted(abs(s["rul_hours"] - s["truth_rul_thr"]) / s["truth_rul_thr"]
                  for s in win if s["rul_hours"] is not None
                  and s["truth_rul_thr"])
    score_err = [abs(s["health_score"] - 100.0 * s["truth_h"]) for s in win]
    return {
        "n": n,
        "stage_normal_pct": round(100.0 * stage_normal / n, 1),
        "rul_not_estimable_pct": round(100.0 * not_estimable / n, 1),
        "rul_zero_output_pct": round(100.0 * zero_rul / n, 1),
        "rul_median_rel_err_pct": (round(100.0 * errs[len(errs) // 2], 1)
                                   if errs else None),
        "mean_abs_score_err": round(sum(score_err) / n, 2),
    }


def rul_stats(samples: list) -> dict:
    """RUL 误差与置信区间覆盖率(主口径: 至失效阈值; 参考口径: 至寿命终点)。

    附带两个运维视角指标:
    - near_failure: 真值 RUL ≤ 72h(告警引擎的预测预警窗口)子集的误差,
      该区间是 RUL 驱动检修决策的实际工作区;
    - zero_output_tail: 输出 0.0h(已判失效)样本中最大的真值 RUL——
      衡量"评分提前跌破阈值导致 RUL 提前归零"的尾巴长度。
    """
    near_failure_hours = 72.0

    def collect(key):
        rel, rel_near, abs_near, abs_h, covered, widths = [], [], [], [], 0, []
        n_ci = 0
        zero_tail_hours = 0.0
        over_ratio = 0.0
        for s in samples:
            truth = s[key]                     # 真值 RUL, 已统一为小时(量纲契约)
            if s["rul_hours"] is None or not truth:
                continue
            err = abs(s["rul_hours"] - truth)
            rel.append(err / truth)
            abs_h.append(err)
            if s["rul_hours"] > truth:
                over_ratio = max(over_ratio, s["rul_hours"] / truth)
            if truth <= near_failure_hours:
                rel_near.append(err / truth)
                abs_near.append(err)
            if s["rul_hours"] == 0.0:
                zero_tail_hours = max(zero_tail_hours, truth)
            if s["ci_low"] is not None:
                n_ci += 1
                widths.append(s["ci_high"] - s["rul_hours"])
                if s["ci_low"] <= truth <= s["ci_high"]:
                    covered += 1
        rel.sort()
        rel_near.sort()
        abs_near.sort()
        abs_h.sort()
        n = len(rel)
        return {
            "n": n,
            "median_rel_err_pct": round(100.0 * rel[n // 2], 1) if n else None,
            "mean_rel_err_pct": (round(100.0 * sum(rel) / n, 1) if n else None),
            "p90_rel_err_pct": (round(100.0 * rel[int(0.9 * n)], 1) if n else None),
            "max_overestimate_ratio": round(over_ratio, 1) if n else None,
            "mae_hours": round(sum(abs_h) / n, 1) if n else None,
            "median_abs_err_hours": round(abs_h[n // 2], 1) if n else None,
            "near_failure_median_rel_err_pct": (round(100.0 * rel_near[len(rel_near) // 2], 1)
                                                if rel_near else None),
            "near_failure_median_abs_err_hours": (round(abs_near[len(abs_near) // 2], 1)
                                                  if abs_near else None),
            "near_failure_n": len(rel_near),
            "zero_output_max_truth_hours": round(zero_tail_hours, 1),
            "ci_n": n_ci,
            "ci_coverage_pct": round(100.0 * covered / n_ci, 1) if n_ci else None,
            "ci_median_half_width_hours": (round(sorted(widths)[n_ci // 2], 1)
                                           if n_ci else None),
        }
    return {"vs_failure_threshold": collect("truth_rul_thr"),
            "vs_end_of_life": collect("truth_rul_eol")}


def calibration_table(samples: list) -> list:
    """健康评分-真值校准表(按真值健康度十分位分箱)。"""
    bins = [(1.0, 0.9), (0.9, 0.8), (0.8, 0.7), (0.7, 0.6), (0.6, 0.5),
            (0.5, 0.4), (0.4, 0.3), (0.3, 0.2), (0.2, 0.1), (0.1, 0.0)]
    rows = []
    for hi, lo in bins:
        sub = [s for s in samples if lo < s["truth_h"] <= hi]
        if not sub:
            continue
        mean_score = sum(s["health_score"] for s in sub) / len(sub)
        err = [abs(s["health_score"] - 100.0 * s["truth_h"]) for s in sub]
        rows.append({"truth_bin": "(%.1f, %.1f]" % (lo, hi), "n": len(sub),
                     "mean_score": round(mean_score, 1),
                     "target": round(100.0 * (lo + hi) / 2, 1),
                     "mean_abs_err": round(sum(err) / len(err), 2)})
    return rows


def per_device_accuracy(samples: list) -> list:
    rows = []
    for did in sorted({s["device_id"] for s in samples}):
        sub = [s for s in samples if s["device_id"] == did]
        acc = sum(1 for s in sub if s["stage"] == s["truth_label"]) / len(sub)
        rows.append({"device_id": did, "n": len(sub),
                     "stage_accuracy": round(acc, 4)})
    return rows


# ------------------------------------------------------------------------------
# 工件渲染
# ------------------------------------------------------------------------------
def render_report(metrics: dict, seed: int, csv_path: str, backend: str) -> str:
    cm = metrics["confusion"]
    md = ["# AI 预测引擎定量评估报告", "",
          "- 生成时间: %s" % metrics["generated_at"],
          "- 复现命令: `python ai-prediction/evaluate.py`(默认种子 %d)"
          % metrics["seed"],
          "- 训练集: `%s`(模拟器 seed=2026 生成的历史数据集)"
          % os.path.relpath(csv_path, PROJECT_ROOT),
          "- 评估集: 独立种子(%d)仿真的三台设备全生命周期轨迹(留出法, 与训练集无样本重叠)"
          % metrics["seed"],
          "- 推理后端: %s(sklearn 缺失时为 numpy-fallback 降级路径)" % backend,
          "- 评估对象: 规则/统计基线(健康评分 + 阶段分类 + RUL 趋势外推), 未训练深度模型",
          "- RUL 量纲口径: 真值与预测/置信区间统一为小时——本评估逐循环喂入"
          "(1 条记录 = 1 tick = 10 分钟), 真值经 truth_rul_hours() 换算; "
          "看板流水线自 v1.3 起同样逐循环喂入(复审报告 07-P1 口径统一), "
          "同一轨迹上在线 RUL 与本评估逐位一致、预测预警触发与真值对齐",
          "",
          "> **局限声明**: 特征与标签同源于自仿真数据(同一健康度真值同时决定"
          "传感器读数与标签), 以下指标系统性偏高, 仅用于回归对比与缺陷验证,"
          "不代表真实产线精度。", ""]
    md += ["## 1. 阶段分类混淆矩阵(行=真值标签, 列=预测阶段)", "",
           "| 真值\\预测 | normal | warning | fault |",
           "| --- | --- | --- | --- |"]
    for t in STAGES:
        row = cm["matrix"][t]
        md.append("| **%s** | %d | %d | %d |" % (t, row["normal"], row["warning"], row["fault"]))
    md += ["", "- 总样本: %d, 准确率: **%.1f%%**, 宏 F1: %.3f"
           % (cm["n"], 100.0 * cm["accuracy"], cm["macro_f1"])]
    for st in STAGES:
        pc = cm["per_class"][st]
        md.append("- %s: precision=%.3f, recall=%.3f, f1=%.3f" % (st, pc["precision"], pc["recall"], pc["f1"]))
    md.append("")
    md += ["| 设备 | 样本 | 阶段准确率 |", "| --- | --- | --- |"]
    for row in metrics["per_device"]:
        md.append("| %s | %d | %.1f%%" % (row["device_id"], row["n"],
                                          100.0 * row["stage_accuracy"]))
    md.append("")
    md += ["## 2. 健康评分-真值校准(校准评分 ≈ 100×真值健康度)", "",
           "| 真值健康度分箱 | 样本 | 平均校准评分 | 目标 | 平均绝对偏差 |",
           "| --- | --- | --- | --- | --- |"]
    for row in metrics["calibration"]:
        md.append("| %s | %d | %.1f | %.1f | %.2f"
                  % (row["truth_bin"], row["n"], row["mean_score"],
                     row["target"], row["mean_abs_err"]))
    md.append("")
    gw = metrics["golden_window"]
    md += ["## 3. 黄金窗口专项(真值健康度 0.6~0.8)", "",
           "| 指标 | 数值 |", "| --- | --- |",
           "| 窗口样本数 | %d |" % gw["n"],
           "| 阶段误判为 normal(漏检) | %.1f%% |" % gw["stage_normal_pct"],
           "| RUL 不可估占比 | %.1f%% |" % gw["rul_not_estimable_pct"],
           "| RUL 输出 0.0h 占比 | %.1f%% |" % gw["rul_zero_output_pct"],
           "| 窗口内 RUL 中位相对误差 | %s |"
           % (("%.1f%%" % gw["rul_median_rel_err_pct"])
              if gw["rul_median_rel_err_pct"] is not None else "-"),
           "| 窗口内评分平均绝对偏差 | %.2f 分 |" % gw["mean_abs_score_err"],
           "",
           "> 窗口内阶段误判集中在真值健康度 0.75~0.8 区间: 该区间传感器信号"
           "仅约 1σ, 与运行工况噪声同量级, 属特征信息量决定的统计下限。", ""]
    thr0 = metrics["rul"]["vs_failure_threshold"]
    md += ["RUL 误差解读: 健康度按幂函数先缓后急退化, 线性外推在退化前半程"
           "系统性高估(凸性偏差)——v1.2 量纲修正后该早期高估如实体现在平均"
           "/P90 相对误差与最大高估倍数上(主口径单点最高 %.1f 倍), 不再被 "
           "tick/小时混用伪装成接近完美; 且相对误差指标以真值 RUL 为分母、"
           "晚期样本分母极小, 全周期中位相对误差因此天然偏高。晚期误差主要"
           "来自代理评分噪声(故障冲击特征使振动 z 波动达 ±4σ 以上)被浅斜率"
           "放大, 属特征信息量决定的方法下限。主口径 80%% CI 实测覆盖率 %s%%, "
           "仍远低于名义 80%%——残差自相关使独立性假设低估长程不确定性, 叠加"
           "凸性高估; 该基线适用于趋势参考与回归对比, 不应作为精确检修时刻"
           "依据; docs §15 已将\"一次趋势外推不处理退化拐点\"列为已知边界。"
           % (thr0["max_overestimate_ratio"], thr0["ci_coverage_pct"]), ""]
    for title, key in (("主口径: RUL 真值 = 真值健康度首次跌破 0.35(失效阈值"
                        " 35 分的校准等价点)的剩余时长", "vs_failure_threshold"),
                       ("参考口径: RUL 真值 = 至寿命终点(健康度=0)的剩余时长",
                        "vs_end_of_life")):
        st = metrics["rul"][key]
        md += ["### %s" % title, "",
               "| 指标 | 数值 |", "| --- | --- |",
               "| 可估样本 | %d |" % st["n"],
               "| RUL 中位相对误差 | %s |" % (("%.1f%%" % st["median_rel_err_pct"]) if st["median_rel_err_pct"] is not None else "-"),
               "| RUL 平均相对误差 | %s |" % (("%.1f%%" % st["mean_rel_err_pct"]) if st["mean_rel_err_pct"] is not None else "-"),
               "| RUL P90 相对误差 | %s |" % (("%.1f%%" % st["p90_rel_err_pct"]) if st["p90_rel_err_pct"] is not None else "-"),
               "| 单点最大高估倍数(预测/真值) | %s |"
               % (("%.1f" % st["max_overestimate_ratio"])
                  if st["max_overestimate_ratio"] else "-"),
               "| RUL MAE / 中位绝对误差 | %s / %s 小时 |"
               % (st["mae_hours"] if st["mae_hours"] is not None else "-",
                  st["median_abs_err_hours"] if st["median_abs_err_hours"] is not None else "-"),
               "| 近失效窗口(真值 RUL ≤ 72h)中位相对/绝对误差 | %s / %s 小时(共 %d 样本) |"
               % (("%.1f%%" % st["near_failure_median_rel_err_pct"])
                  if st["near_failure_median_rel_err_pct"] is not None else "-",
                  st["near_failure_median_abs_err_hours"]
                  if st["near_failure_median_abs_err_hours"] is not None else "-",
                  st["near_failure_n"]),
               "| 提前归零尾巴(输出 0.0h 样本的最大真值 RUL) | %s 小时 |"
               % (st["zero_output_max_truth_hours"]
                  if st["zero_output_max_truth_hours"] is not None else "-"),
               "| 80%% 置信区间实测覆盖率 | %s(共 %d 个带 CI 样本) |"
               % (("%.1f%%" % st["ci_coverage_pct"]) if st["ci_coverage_pct"] is not None else "-",
                  st["ci_n"]),
               "| 置信区间半宽中位数 | %s 小时 |"
               % (st["ci_median_half_width_hours"] if st["ci_median_half_width_hours"] is not None else "-"),
               ""]
    md += ["## 4. 复现方式", "",
           "```bash",
           "# 训练数据集(已提交, 可选重新生成, seed=2026 保证一致)",
           "python data-ingestion/data_simulator.py --mode csv --rows-per-device 220",
           "# 运行本评估(默认种子 7, 与 predictive_engine.py --demo 一致)",
           "python ai-prediction/evaluate.py",
           "# RUL 置信区间单元用例(快/慢退化两场景)",
           "python ai-prediction/evaluate.py --selftest",
           "```", ""]
    return "\n".join(md)


# ------------------------------------------------------------------------------
# RUL 置信区间单元用例(快/慢退化两场景)
# ------------------------------------------------------------------------------
def _synth_series(rng, b: float, sigma: float, n: int = 40, y0: float = None):
    """生成线性退化 + 高斯噪声的健康评分序列, 返回 (序列, 真值失效时刻小时)。"""
    if y0 is None:
        y0 = 100.0 + b * n / 2.0          # 使窗口末点稳定在 y0
    y = [y0 + b * i + rng.normal(0.0, sigma) for i in range(n)]
    current = y0 + b * (n - 1)
    true_rul_hours = max(0.0, (current - 35.0) / (-b)) * TICK_HOURS
    return y, true_rul_hours


# ------------------------------------------------------------------------------
# 喂入口径一致性用例(v1.3 统一, 复审报告 07-N-P1-1)
# ------------------------------------------------------------------------------
def _synth_linear_exact(b: float, n: int = 200):
    """无噪声线性退化评分序列(确定性): 每条 = 1 tick, 真值失效时刻解析可得。"""
    y = [100.0 + b * i for i in range(n)]
    true_hours = (y[-1] - 35.0) / (-b) * TICK_HOURS
    return y, true_hours


def _synth_convex_exact(n: int, life: int):
    """幂函数凸性退化评分序列(确定性): score_i = 100×(1-(i/life)^1.6),
    与数据模拟器 DeviceSimulator.health() 的衰退曲线同族。"""
    return [100.0 * (1.0 - (i / life) ** 1.6) for i in range(n)]


def run_caliber_selftest() -> bool:
    """喂入口径回归锁(2026-09 第四轮修补, 复审报告 07-N-P1-1)。

    背景: v1.2 及之前看板流水线每 6 个模拟循环只把最后 1 条喂给引擎, 而
    RULPredictor 按"1 条 = 1 tick"换算, 同一设备状态在线 RUL 与评估口径
    存在约 3 倍系统性分歧、预测预警提前触发; v1.3 起看板逐循环喂入, 与本
    评估共用同一换算。两条确定性断言防止口径再漂移:
      1. 逐 tick 口径锁: 无噪线性序列上 predict() 必须还原解析真值
         (换算常量/公式漂移会立即失败);
      2. 6:1 节流危害用例: 同一凸性退化序列, 若按 6:1 抽样喂入而仍用默认
         "每条 = 1 tick"语义(v1.2 看板行为), 换算差 1/6 与凸性窗口效应
         (平均斜率被低估)叠加, RUL 系统性失真——实测约为逐 tick 口径的
         0.25 倍, 预警因此大幅提前触发。断言其偏离逐 tick 口径 10% 以上,
         锁定"禁止节流喂入"这一设计决定。
    """
    pred = RULPredictor()
    ok = True
    # 1) 逐 tick 口径锁
    y, true_hours = _synth_linear_exact(b=-0.02, n=200)
    r = pred.predict(y)
    rel = abs(r["rul_hours"] - true_hours) / true_hours
    lock_ok = rel <= 0.01
    ok = ok and lock_ok
    print("[%s] 逐tick口径锁: 线性序列 RUL=%.1fh(解析真值 %.1fh, 偏差 %.3f%%)"
          % ("PASS" if lock_ok else "FAIL", r["rul_hours"], true_hours, 100.0 * rel))
    # 2) 6:1 节流危害用例(旧看板行为的反例)
    y_c = _synth_convex_exact(n=764, life=1320)
    r_tick = pred.predict(y_c)
    r_thr = pred.predict(y_c[::6])
    ratio = r_thr["rul_hours"] / r_tick["rul_hours"]
    hazard_ok = 0.10 <= ratio <= 0.60
    ok = ok and hazard_ok
    print("[%s] 6:1节流危害锁: 逐tick喂入 RUL=%.1fh, 节流喂入 RUL=%.1fh"
          "(比值 %.2f, 复现 v1.2 的约 1/4 失真, 须偏离逐tick口径 10%% 以上)"
          % ("PASS" if hazard_ok else "FAIL", r_tick["rul_hours"],
             r_thr["rul_hours"], ratio))
    for name, bad in (("逐tick口径锁", not lock_ok),
                      ("6:1节流危害锁", not hazard_ok)):
        if bad:
            print("       未通过: %s" % name)
    return ok


def run_selftest() -> bool:
    """RUL 置信区间量纲单元用例 + 喂入口径一致性回归锁。

    慢退化(|b| 小)是 v1.0 量纲错误的重灾区: margin 未除以 |b| 时 CI 被
    系统性压窄, 无法覆盖真值。两用例断言:
      1. 点估计接近真值(容差由斜率估计方差决定);
      2. 真值失效时刻落在 80% CI 内(覆盖率性质);
      3. CI 宽度与外推不确定性相称(慢退化宽、快退化相对窄)。
    另含真值换算契约用例: truth_rul_hours() 必须把 tick 差值换算为小时
    (v1.0 曾把 tick 差值直接当小时比较, 复审报告 07-N-P0-1), 以及
    TICK_HOURS 与模拟器 SIM_TICK_MINUTES 的跨模块一致性;
    run_caliber_selftest() 锁定"逐 tick 喂入"口径(复审报告 07-N-P1-1)。
    """
    pred = RULPredictor()
    rng = __import__("numpy").random.RandomState(2026)
    cases = [
        # (名称, 斜率 分/tick, 噪声 sigma, 点估计容差比例, CI 半宽下限比例, CI 半宽上限比例)
        ("快退化", -0.50, 0.30, 0.35, 0.0, 0.60),
        ("慢退化", -0.01, 0.30, 0.50, 0.25, None),
    ]
    ok = True
    for name, b, sigma, tol, w_lo, w_hi in cases:
        y, true_hours = _synth_series(rng, b, sigma)
        r = pred.predict(y)
        est = r["rul_hours"]
        rel = abs(est - true_hours) / true_hours
        half_width = (r["rul_ci_high"] - r["rul_ci_low"]) / 2.0
        covered = r["rul_ci_low"] <= true_hours <= r["rul_ci_high"]
        checks = {
            "点估计": rel <= tol,
            "CI覆盖真值": covered,
            "CI下界>=0": r["rul_ci_low"] >= 0.0,
            "CI宽度下限": w_lo is None or half_width >= w_lo * est,
            "CI宽度上限": w_hi is None or half_width <= w_hi * est,
        }
        case_ok = all(checks.values())
        ok = ok and case_ok
        print("[%s] %s场景: RUL=%.1fh(真值 %.1fh) CI=[%.1f, %.1f] 半宽=%.1fh (%.0f%% RUL)"
              % ("PASS" if case_ok else "FAIL", name, est, true_hours,
                 r["rul_ci_low"], r["rul_ci_high"], half_width,
                 100.0 * half_width / est))
        for k, v in checks.items():
            if not v:
                print("       未通过: %s" % k)
    # 真值换算量纲契约: 60 tick × (10 分钟/tick) 必须等于 10 小时,
    # 且 RULPredictor 点估计与真值同量纲(上例 rel 误差已按小时算);
    # TICK_HOURS 须与模拟器的 SIM_TICK_MINUTES 保持跨模块一致(唯一换算源)
    dim_checks = {
        "tick→小时换算": abs(truth_rul_hours(0, 60) - 10.0) < 1e-9,
        "负差值截断为0": truth_rul_hours(70, 60) == 0.0,
        "TICK_HOURS<1h": 0.0 < TICK_HOURS < 1.0,
        "与SIM_TICK_MINUTES一致": abs(TICK_HOURS - SIM_TICK_MINUTES / 60.0) < 1e-12,
    }
    dim_ok = all(dim_checks.values())
    ok = ok and dim_ok
    print("[%s] 真值RUL量纲契约: 60tick=%.1fh(期望10.0h), TICK_HOURS=%.4f"
          "(= SIM_TICK_MINUTES %d/60)"
          % ("PASS" if dim_ok else "FAIL", truth_rul_hours(0, 60), TICK_HOURS,
             SIM_TICK_MINUTES))
    for k, v in dim_checks.items():
        if not v:
            print("       未通过: %s" % k)
    # 喂入口径一致性回归锁(v1.3 统一, 复审报告 07-N-P1-1)
    ok = run_caliber_selftest() and ok
    print("\nRUL CI 与喂入口径自检结果: %s" % ("全部通过" if ok else "存在失败"))
    return ok


# ------------------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="数字孪生平台 - AI 预测引擎定量评估")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="训练用历史数据集路径")
    parser.add_argument("--seed", type=int, default=7,
                        help="评估轨迹随机种子(默认 7, 与 --demo 一致)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="工件输出目录")
    parser.add_argument("--no-write", action="store_true", help="仅打印, 不写工件")
    parser.add_argument("--selftest", action="store_true",
                        help="RUL 置信区间单元用例(快/慢退化)")
    args = parser.parse_args()
    if args.selftest:
        sys.exit(0 if run_selftest() else 1)

    if not os.path.exists(args.csv):
        print("[评估] 未找到历史数据集 %s, 先生成 ..." % args.csv)
        from data_simulator import generate_historical_csv
        generate_historical_csv(args.csv)

    samples, backend = run_trajectories(args.seed, args.csv)
    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "csv": args.csv,
        "backend": backend,
        "confusion": confusion_matrix(samples),
        "per_device": per_device_accuracy(samples),
        "calibration": calibration_table(samples),
        "golden_window": golden_window_stats(samples),
        "rul": rul_stats(samples),
    }

    cm = metrics["confusion"]
    gw = metrics["golden_window"]
    thr = metrics["rul"]["vs_failure_threshold"]
    print("\n[评估] 阶段分类: 准确率 %.1f%%, 宏F1 %.3f (n=%d)"
          % (100.0 * cm["accuracy"], cm["macro_f1"], cm["n"]))
    print("[评估] 黄金窗口(0.6~0.8): 漏检 normal %.1f%%, RUL 不可估 %.1f%%"
          % (gw["stage_normal_pct"], gw["rul_not_estimable_pct"]))
    print("[评估] RUL(至失效阈值): 中位相对误差 %s%%, MAE %s h, 80%%CI 覆盖率 %s%%"
          % (thr["median_rel_err_pct"], thr["mae_hours"], thr["ci_coverage_pct"]))

    if not args.no_write:
        os.makedirs(args.out_dir, exist_ok=True)
        md_path = os.path.join(args.out_dir, "evaluation_report.md")
        json_path = os.path.join(args.out_dir, "evaluation_metrics.json")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(render_report(metrics, args.seed, args.csv, metrics["backend"]))
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print("[评估] 工件已写入: %s / %s" % (md_path, json_path))


if __name__ == "__main__":
    main()
