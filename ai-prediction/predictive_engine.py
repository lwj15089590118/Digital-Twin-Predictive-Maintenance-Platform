# -*- coding: utf-8 -*-
"""
================================================================================
 AI 预测引擎 (Predictive Engine)
================================================================================
模块职责:
    1. 异常检测: 基于历史正常运行数据建立各传感器通道的基线分布,
       使用 IsolationForest(sklearn 可用时)或 Mahalanobis 距离(纯 numpy
       回退)识别实时数据中的异常模式;
    2. 健康评分: 将多通道偏差加权融合为 0~100% 的设备健康评分;
    3. 剩余寿命预测 (RUL): 对近期健康评分序列拟合退化趋势线, 外推至
       失效阈值, 换算剩余运行时长, 并基于拟合残差给出 80% 置信区间;
    4. 劣化阶段分类: 使用随机森林(sklearn 可用时)或规则引擎(回退)判断
       设备当前所处生命周期阶段。

设计说明:
    - sklearn 缺失时自动降级为纯 numpy 统计实现, 保证任何环境可运行;
    - predict() 不使用模拟器的真值 health 字段, 仅凭传感器读数推理,
       与真实工业场景一致。

运行示例:
    python predictive_engine.py --train ../data-ingestion/historical_data.csv
    python predictive_engine.py --demo
================================================================================
"""

import argparse
import csv
import json
import math
import os
from datetime import datetime

import numpy as np

try:                                      # sklearn 为可选依赖: 有则增强, 无则降级
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(PROJECT_ROOT, "data-ingestion", "historical_data.csv")
ENGINE_STATE_PATH = os.path.join(PROJECT_ROOT, "data", "ai", "engine_state.json")

# 传感器通道与工程单位
CHANNELS = ["vibration", "temperature", "current", "voltage", "rpm"]
# 健康评分中各通道的权重(经验值: 振动是机械故障最灵敏的指示器)
CHANNEL_WEIGHTS = {"vibration": 0.35, "temperature": 0.25,
                   "current": 0.20, "voltage": 0.10, "rpm": 0.10}
# RUL 失效阈值: 健康评分低于该值认为功能性失效, 必须停机检修
FAILURE_THRESHOLD = 35.0
# 每个采样 tick 对应的真实运行时长(小时), 与数据模拟器保持一致
TICK_HOURS = 10.0 / 60.0
# 参与 RUL 拟合的近期窗口大小(条)
RUL_WINDOW = 40


# ------------------------------------------------------------------------------
# 历史数据加载(csv 模块实现, 不依赖 pandas)
# ------------------------------------------------------------------------------
def load_history(csv_path: str) -> list:
    """读取历史数据集 CSV 为记录列表(每条为 dict, 数值字段已转 float)。"""
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec = {"timestamp": row.get("timestamp", ""),
                   "device_id": row.get("device_id", ""),
                   "label": row.get("label", ""),
                   "cycle": int(float(row.get("cycle", 0)))}
            for ch in CHANNELS:
                rec[ch] = float(row.get(ch, 0.0) or 0.0)
            # health 真值列可能缺失(实时数据无真值), 缺省置 None
            h = row.get("health")
            rec["health_truth"] = float(h) if h not in (None, "") else None
            records.append(rec)
    return records


# ------------------------------------------------------------------------------
# 基线统计模型: 学习"健康状态"下各通道的分布
# ------------------------------------------------------------------------------
class BaselineProfile:
    """单台设备健康基线: 各通道均值/标准差/容差上界。

    健康样本选取策略: 优先取 label=normal 的样本; 若无标签则取
    健康度最高的前 25% 样本(早期数据通常接近健康)。
    """

    def __init__(self):
        self.mean = {}
        self.std = {}
        self.n_samples = 0

    def fit(self, records: list):
        """从记录列表中筛选健康样本并估计基线分布。"""
        healthy = [r for r in records if r.get("label") == "normal"]
        if not healthy:
            # 无标签回退: 取振动最小的前 25% 样本近似视为健康
            healthy = sorted(records, key=lambda r: r["vibration"])
            healthy = healthy[:max(5, len(records) // 4)]
        self.n_samples = len(healthy)
        for ch in CHANNELS:
            values = np.array([r[ch] for r in healthy], dtype=float)
            self.mean[ch] = float(np.mean(values))
            self.std[ch] = float(np.std(values)) or 1e-3     # 防止除零

    def z_scores(self, record: dict) -> dict:
        """计算一条记录相对基线的 z 分数(标准化偏差)。"""
        return {ch: (record.get(ch, self.mean[ch]) - self.mean[ch]) / self.std[ch]
                for ch in CHANNELS}

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std, "n_samples": self.n_samples}


# ------------------------------------------------------------------------------
# 异常检测器: IsolationForest(主) / Mahalanobis(回退)
# ------------------------------------------------------------------------------
class AnomalyDetector:
    """异常检测: 输出 0~100 的异常分(越高越异常)。"""

    def __init__(self):
        self.method = "mahalanobis"                # 实际采用的方法(训练后确定)
        self.model = None                          # sklearn 模型(若可用)
        self.cov_inv = None                        # 协方差逆矩阵(numpy 回退用)
        self.center = None

    def fit(self, records: list, baseline: BaselineProfile):
        """用健康样本训练异常检测模型。"""
        healthy = [r for r in records if r.get("label") == "normal"]
        if not healthy:
            healthy = sorted(records, key=lambda r: r["vibration"])[:max(5, len(records) // 4)]
        X = np.array([[r[ch] for ch in CHANNELS] for r in healthy], dtype=float)
        if SKLEARN_AVAILABLE and len(X) >= 20:
            # IsolationForest: 无监督孤点检测, 对多维联合分布异常敏感
            self.model = IsolationForest(n_estimators=120, contamination=0.02,
                                         random_state=42)
            self.model.fit(X)
            self.method = "isolation_forest"
        else:
            # 纯 numpy 回退: 各通道 z 分数的马氏距离(协方差在标准化空间估计)
            self.center = np.array([baseline.mean[ch] for ch in CHANNELS])
            self.scale = np.array([baseline.std[ch] for ch in CHANNELS])
            Z = (X - self.center) / self.scale
            cov = np.cov(Z.T) + np.eye(len(CHANNELS)) * 1e-3
            self.cov_inv = np.linalg.inv(cov)
            self.method = "mahalanobis"

    def score(self, record: dict) -> float:
        """返回 0(完全正常)~100(高度异常) 的异常分。"""
        x = np.array([record.get(ch, 0.0) for ch in CHANNELS], dtype=float)
        if self.method == "isolation_forest":
            # decision_function: 越小越异常, 经 sigmoid 映射到 0~100
            raw = self.model.decision_function(x.reshape(1, -1))[0]
            return float(np.clip(100.0 / (1.0 + math.exp(raw / 0.08)), 0.0, 100.0))
        # Mahalanobis 距离: 先标准化到与协方差一致的尺度再算距离,
        # d<=1 视为正常区, 之后每 +1σ 距离累加 15 分
        z = (x - self.center) / self.scale
        d = math.sqrt(float(z @ self.cov_inv @ z))
        return float(min(100.0, max(0.0, (d - 1.0) * 15.0)))


# ------------------------------------------------------------------------------
# 健康评分模型: 多通道加权融合
# ------------------------------------------------------------------------------
class HealthScorer:
    """将实时记录转化为 0~100% 健康评分。

    评分逻辑: 对每个通道计算"恶化度"(0~1, 方向感知: 振动/温度/电流
    单边上升恶化, 电压/转速双边偏离恶化), 按通道权重加权求和后映射
    到 100*(1-deterioration)。
    """

    # 单边恶化通道(只有超过基线才算恶化): 振动/温度/电流
    ONE_SIDED = {"vibration", "temperature", "current"}

    def __init__(self, baseline: BaselineProfile):
        self.baseline = baseline

    def channel_deterioration(self, ch: str, record: dict) -> float:
        """计算单通道恶化度: 0=完全健康, 1=严重恶化。"""
        z = self.baseline.z_scores(record)[ch]
        if ch in self.ONE_SIDED:
            # 单边: z<=1 视为健康; z 每 +3 sigma 恶化度 +0.5
            return float(np.clip((z - 1.0) / 6.0, 0.0, 1.0))
        # 双边通道(电压/转速): |z| 偏离即恶化
        return float(np.clip((abs(z) - 2.0) / 6.0, 0.0, 1.0))

    def score(self, record: dict) -> float:
        """加权健康评分(0~100, 保留 1 位小数)。"""
        det = sum(CHANNEL_WEIGHTS[ch] * self.channel_deterioration(ch, record)
                  for ch in CHANNELS)
        return round(100.0 * (1.0 - det), 1)


# ------------------------------------------------------------------------------
# RUL 剩余寿命预测器: 趋势外推 + 置信区间
# ------------------------------------------------------------------------------
class RULPredictor:
    """基于健康评分退化趋势的剩余寿命预测。

    方法:
        1. 取最近 RUL_WINDOW 条健康评分序列;
        2. 最小二乘拟合一次退化趋势(每日下降速率), 可选检测加速退化;
        3. 外推至失效阈值 FAILURE_THRESHOLD 得 RUL(小时);
        4. 用拟合残差的标准差构造 80% 置信区间
           (正态近似: 均值 ± 1.2816 * sigma_residual * sqrt(外推步数比))。
    """

    def predict(self, health_series: list) -> dict:
        """对健康评分序列(按时间升序, 0~100)预测 RUL。

        Returns:
            {rul_hours, rul_ci_low, rul_ci_high, decline_per_day, trend, method}
            trend: rising/stable/degrading/rapid_degrading
        """
        if len(health_series) < 8:
            return {"rul_hours": None, "rul_ci_low": None, "rul_ci_high": None,
                    "decline_per_day": None, "trend": "insufficient_data",
                    "method": "trend_extrapolation"}
        y = np.array(health_series[-RUL_WINDOW:], dtype=float)
        n = len(y)
        x = np.arange(n, dtype=float)
        # ---- 最小二乘一次拟合: y = a + b*x ----
        A = np.vstack([np.ones(n), x]).T
        coef, res, _, _ = np.linalg.lstsq(A, y, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        # 每日下降速率: 每 tick = TICK_HOURS 小时
        decline_per_day = -b * 24.0 / TICK_HOURS
        # 拟合残差标准差(置信区间宽度来源)
        y_hat = A @ coef
        sigma = float(np.sqrt(np.sum((y - y_hat) ** 2) / max(1, n - 2)))
        # ---- 趋势判定 ----
        if b >= -0.005:
            trend = "stable" if abs(b) < 0.02 else "rising"
        elif decline_per_day < 12.0:
            trend = "degrading"
        else:
            trend = "rapid_degrading"
        # ---- 外推至失效阈值 ----
        current = float(y[-1])
        if current <= FAILURE_THRESHOLD:
            return {"rul_hours": 0.0, "rul_ci_low": 0.0, "rul_ci_high": 0.0,
                    "decline_per_day": round(decline_per_day, 2),
                    "trend": "failed", "method": "trend_extrapolation"}
        if b >= 0:
            # 评分未下降: 无法外推, 返回保守的大值并标记
            return {"rul_hours": None, "rul_ci_low": None, "rul_ci_high": None,
                    "decline_per_day": round(decline_per_day, 2), "trend": trend,
                    "method": "trend_extrapolation"}
        ticks_to_fail = (current - FAILURE_THRESHOLD) / (-b)
        rul_hours = ticks_to_fail * TICK_HOURS
        # 80% 置信区间: 残差不确定性随外推距离放大(比例误差模型)
        margin = 1.2816 * max(sigma, 0.5) * math.sqrt(max(ticks_to_fail, 1.0))
        ci_low = max(0.0, rul_hours - margin * TICK_HOURS)
        ci_high = rul_hours + margin * TICK_HOURS
        return {"rul_hours": round(rul_hours, 1),
                "rul_ci_low": round(ci_low, 1),
                "rul_ci_high": round(ci_high, 1),
                "decline_per_day": round(decline_per_day, 2),
                "trend": trend, "method": "trend_extrapolation"}


# ------------------------------------------------------------------------------
# 劣化阶段分类器: 随机森林(主) / 规则引擎(回退)
# ------------------------------------------------------------------------------
class StageClassifier:
    """判断设备当前生命周期阶段: normal / warning / fault。"""

    def __init__(self):
        self.model = None
        self.method = "rule_based"

    def fit(self, records: list, baseline: BaselineProfile):
        """用带标签历史数据训练随机森林; 不可用时保持规则模式。"""
        labeled = [r for r in records if r.get("label") in ("normal", "warning", "fault")]
        if SKLEARN_AVAILABLE and len(labeled) >= 30:
            X = np.array([[r[ch] for ch in CHANNELS] for r in labeled], dtype=float)
            y = np.array([1 if r["label"] == "normal" else
                          (2 if r["label"] == "warning" else 3) for r in labeled])
            self.model = RandomForestClassifier(n_estimators=100, max_depth=8,
                                                random_state=42)
            self.model.fit(X, y)
            self.method = "random_forest"

    def predict(self, record: dict, health_score: float) -> str:
        """返回预测标签 normal/warning/fault。"""
        if self.method == "random_forest" and self.model is not None:
            x = np.array([[record[ch] for ch in CHANNELS]], dtype=float)
            code = int(self.model.predict(x)[0])
            return {1: "normal", 2: "warning", 3: "fault"}[code]
        # 规则回退: 直接以健康评分阈值划分
        if health_score > 80.0:
            return "normal"
        return "warning" if health_score > 50.0 else "fault"


# ------------------------------------------------------------------------------
# 预测引擎编排器
# ------------------------------------------------------------------------------
class PredictiveEngine:
    """整合基线/异常检测/健康评分/RUL/阶段分类的统一推理入口。"""

    def __init__(self):
        self.baselines = {}         # device_id -> BaselineProfile
        self.detectors = {}         # device_id -> AnomalyDetector
        self.scorers = {}           # device_id -> HealthScorer
        self.classifiers = {}       # device_id -> StageClassifier
        self.health_history = {}    # device_id -> 最近健康评分序列
        self.trained_at = None
        self.backend = "numpy-fallback"

    # ---------------------------- 训练 ----------------------------
    def train(self, csv_path: str = DEFAULT_CSV) -> dict:
        """用历史数据集训练全部子模型, 返回训练摘要。"""
        records = load_history(csv_path)
        devices = sorted({r["device_id"] for r in records})
        summary = {"devices": {}, "total_samples": len(records),
                   "sklearn": SKLEARN_AVAILABLE, "csv": csv_path}
        for did in devices:
            subset = [r for r in records if r["device_id"] == did]
            baseline = BaselineProfile()
            baseline.fit(subset)
            detector = AnomalyDetector()
            detector.fit(subset, baseline)
            scorer = HealthScorer(baseline)
            clf = StageClassifier()
            clf.fit(subset, baseline)
            self.baselines[did] = baseline
            self.detectors[did] = detector
            self.scorers[did] = scorer
            self.classifiers[did] = clf
            self.health_history[did] = []
            summary["devices"][did] = {
                "baseline_samples": baseline.n_samples,
                "anomaly_method": detector.method,
                "stage_method": clf.method,
                "baseline_mean": {k: round(v, 2) for k, v in baseline.mean.items()},
            }
        self.trained_at = datetime.now().isoformat(timespec="seconds")
        self.backend = "sklearn" if SKLEARN_AVAILABLE else "numpy-fallback"
        self.save_state()
        return summary

    # ---------------------------- 实时推理 ----------------------------
    def assess(self, record: dict) -> dict:
        """对一条实时记录做完整评估。

        Returns:
            {device_id, timestamp, health_score, anomaly_score, stage,
             rul: {...}, features: {z_scores}, backend}
        """
        did = record.get("device_id")
        if did not in self.baselines:
            raise KeyError("设备 %s 未训练, 请先调用 train()" % did)
        scorer = self.scorers[did]
        health_score = scorer.score(record)
        anomaly_score = round(self.detectors[did].score(record), 1)
        stage = self.classifiers[did].predict(record, health_score)
        # 维护健康评分历史(供 RUL 拟合)
        self.health_history.setdefault(did, []).append(health_score)
        if len(self.health_history[did]) > 500:
            self.health_history[did] = self.health_history[did][-500:]
        rul = RULPredictor().predict(self.health_history[did])
        z = self.baselines[did].z_scores(record)
        return {
            "device_id": did,
            "timestamp": record.get("timestamp"),
            "health_score": health_score,
            "anomaly_score": anomaly_score,
            "stage": stage,
            "rul": rul,
            "features": {ch: round(float(z[ch]), 2) for ch in CHANNELS},
            "backend": self.backend,
        }

    # ---------------------------- 持久化 ----------------------------
    def save_state(self, path: str = ENGINE_STATE_PATH):
        """保存基线统计与模型方法标注(sklearn 模型本体不序列化, 重启重训)。"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {"trained_at": self.trained_at, "backend": self.backend,
                 "devices": {did: self.baselines[did].to_dict()
                             for did in self.baselines}}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def report_text(self, result: dict, device_name: str = "") -> str:
        """将评估结果渲染为人可读报告(用于导出与控制台)。"""
        rul = result["rul"]
        lines = [
            "设备智能预测报告".center(46, "="),
            "设备: %s %s" % (result["device_id"], device_name),
            "评估时间: %s" % result["timestamp"],
            "健康评分: %.1f / 100" % result["health_score"],
            "异常评分: %.1f / 100 (方法: %s)" % (
                result["anomaly_score"], self.detectors[result["device_id"]].method),
            "生命周期阶段: %s" % result["stage"],
        ]
        if rul.get("rul_hours") is not None:
            lines.append("剩余寿命 RUL: %.1f 小时 (80%%置信区间: %.1f ~ %.1f)"
                         % (rul["rul_hours"], rul["rul_ci_low"], rul["rul_ci_high"]))
            lines.append("退化速率: %.2f 分/天, 趋势: %s"
                         % (rul["decline_per_day"], rul["trend"]))
        else:
            lines.append("剩余寿命 RUL: 暂不可估(%s)" % rul.get("trend"))
        lines.append("通道偏差(z分数): " + ", ".join(
            "%s=%+.2f" % (k, v) for k, v in result["features"].items()))
        lines.append("=" * 46)
        return "\n".join(lines)


# ------------------------------------------------------------------------------
# 演示与命令行
# ------------------------------------------------------------------------------
def demo():
    """端到端演示: 训练 -> 快进生命周期 -> 逐阶段评估打印报告。"""
    engine = PredictiveEngine()
    print("[预测引擎] 训练中, sklearn 可用: %s ..." % SKLEARN_AVAILABLE)
    summary = engine.train(DEFAULT_CSV)
    for did, info in summary["devices"].items():
        print("  %s 基线样本=%d 异常检测=%s 阶段分类=%s"
              % (did, info["baseline_samples"], info["anomaly_method"], info["stage_method"]))
    # 用模拟器快进生成实时流, 验证评分随退化下降(每轮都评估以累积健康序列)
    import sys
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "data-ingestion"))
    from data_simulator import DataSimulator
    sim = DataSimulator(seed=7)
    checkpoints = {0, 250, 500, 750, 1000, 1250}
    for _ in range(1300):
        for rec in sim.step_all():
            result = engine.assess(rec)
            if rec["cycle"] in checkpoints:
                print(engine.report_text(result, rec["device_name"]))


def main():
    parser = argparse.ArgumentParser(description="数字孪生平台 - AI 预测引擎")
    parser.add_argument("--train", action="store_true", help="训练模型并保存基线")
    parser.add_argument("--demo", action="store_true", help="端到端演示")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="历史数据集路径")
    args = parser.parse_args()
    if args.demo:
        demo()
    elif args.train:
        engine = PredictiveEngine()
        summary = engine.train(args.csv)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
