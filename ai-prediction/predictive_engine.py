# -*- coding: utf-8 -*-
"""
================================================================================
 AI 预测引擎 (Predictive Engine)
================================================================================
模块职责:
    1. 异常检测: 基于历史正常运行数据建立各传感器通道的基线分布,
       使用 IsolationForest(sklearn 可用时)或 Mahalanobis 距离(纯 numpy
       回退)识别实时数据中的异常模式;
    2. 健康评分: 将多通道偏差加权融合为 0~100% 的设备健康评分,
       并经真值校准的分段线性表映射, 使评分 ≈ 100×真实健康度;
    3. 剩余寿命预测 (RUL): 对近期健康评分序列拟合退化趋势线, 外推至
       失效阈值, 换算剩余运行时长, 并基于拟合残差给出 80% 置信区间
       (残差评分点量纲经标准外推方差放大后除以拟合斜率换算为小时);
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
# RUL 失效阈值: 健康评分低于该值认为功能性失效, 必须停机检修。
# 校准评分 ≈ 100×真值健康度, 故 35 分对应真值健康度约 0.35(严重退化区间)
FAILURE_THRESHOLD = 35.0
# RUL 输出上限(小时, 90 天): 退化早期斜率趋近于零, 微小的拟合斜率噪声
# 会让线性外推产生无意义的巨额数值; 按行业惯例对 RUL 做上限截断
# (如 C-MAPSS 基准将 RUL 截断至 125 循环), 超出运维规划视野的预测不再输出
RUL_MAX_HOURS = 2160.0
# 每个采样 tick 对应的真实运行时长(小时), 与数据模拟器保持一致。
# 喂入口径契约(2026-09 第四轮修补, 复审报告 07-N-P1-1): 所有喂入方(看板
# 流水线 / evaluate.py 评估 / 演示)必须把模拟产生的每一条记录逐条喂给引擎,
# 即 1 条记录 = 1 tick = TICK_HOURS 小时——若按 6:1 等节流喂入而仍用本常量
# 换算, 在线 RUL 会与物理时间产生数倍系统性分歧(该 bug 曾使看板 RUL 与
# 评估口径相差约 3 倍、预测预警提前触发)
TICK_HOURS = 10.0 / 60.0
# 参与 RUL 拟合的近期窗口大小(条)。取 144 = 一整天(温度日周期正弦的
# 完整周期, 见 data_simulator.py 环境波动项): 窗口覆盖完整周期后, 日周期
# 在斜率估计中正负抵消——v1.0 的 40 点窗口内日周期伪斜率与真实退化斜率
# 同量级, 是黄金窗口 RUL 大量"不可估"的主因(2026-08 修复审查报告 07)
RUL_WINDOW = 144


# ------------------------------------------------------------------------------
# 历史数据加载(csv 模块实现, 不依赖 pandas)
# ------------------------------------------------------------------------------
def load_history(csv_path: str) -> list:
    """读取历史数据集 CSV 为记录列表(每条为 dict, 数值字段已转 float)。

    缺失必需数值列时直接报错——静默置 0 会训练出垃圾基线且难以排查。
    """
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [ch for ch in CHANNELS if ch not in (reader.fieldnames or [])]
        if missing:
            raise ValueError("历史数据集 %s 缺少必需数值列: %s"
                             % (csv_path, ", ".join(missing)))
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
    """将实时记录转化为 0~100% 健康评分(经真值校准)。

    评分逻辑: 对每个通道计算"恶化度"(0~1, 方向感知: 振动/温度/电流
    单边上升恶化, 电压/转速双边偏离恶化), 按通道权重加权求和得到
    原始分, 再经保序回归校准表映射为校准评分。

    校准说明(2026-08 修复审查报告 07-P1-2):
        v1.0 版本单边通道 z>1σ 起算、每 6σ 记满, 在真值健康度 0.6~0.8
        的"预测性维护黄金窗口"存在死区(实测 40% 样本被误判 normal、
        35% 样本 RUL 不可估)。本次修复分两步:
        1) 去死区: 单边通道 z>0.5σ 起算, 且按通道灵敏度分配满量程跨度
           (振动 42σ/温度与电流 20σ, 双边通道 0.8σ 起算、8σ 记满)——
           跨度过小会让振动(基线 σ 极小、z 爆炸式增长)过早饱和,
           过大会淹没温度/电流的有效信号, 常数经真值上网格搜索确定;
        2) 标度校准: 原始分对真值健康度单调但非线性, 经保序回归
           (isotonic, 分段线性表)映射为校准评分, 使 score ≈ 100×真值
           健康度。映射常数与校准表均在模拟器真值上离线标定——等价于
           工业现场用失效数据离线标定评分模型, 运行时推理不读取真值
           health 字段。校准效果见 reports/evaluation_report.md 第 2 节。
    """

    # 单边恶化通道(只有超过基线才算恶化): 振动/温度/电流
    ONE_SIDED = {"vibration", "temperature", "current"}
    # 原始恶化度映射常数(离线校准, 方法见类文档字符串)
    Z0_ONE_SIDED = 0.5                          # 单边通道起算阈值(σ)
    SPAN_ONE_SIDED = {"vibration": 42.0,        # 振动: 满量程跨度(σ)
                      "temperature": 20.0, "current": 20.0}
    Z0_TWO_SIDED = 0.8                          # 双边通道起算阈值(|z|)
    SPAN_TWO_SIDED = 8.0                        # 双边通道满量程跨度(σ)
    # 保序回归校准表(原始分 -> 校准分), 分段线性插值; 两端越界即截断。
    # 在 4 条独立仿真轨迹(seed=2026/11/22/33, 共约 1.9 万样本)上池化标定,
    # 避免单条轨迹的工况噪声过拟合; 留出种子(7/44/55)验证中位偏差 ~4 分
    SCORE_CALIBRATION = [
        (0.0, 0.0), (30.0, 0.8), (34.7, 5.9), (41.0, 13.2), (47.4, 21.5),
        (56.6, 32.0), (66.1, 41.1), (76.2, 50.3), (86.0, 59.2), (91.8, 66.8),
        (94.8, 73.9), (96.7, 82.6), (97.9, 87.4), (98.6, 91.2), (99.2, 92.5),
        (99.7, 93.7), (100.0, 94.9),
    ]

    def __init__(self, baseline: BaselineProfile):
        self.baseline = baseline

    def channel_deterioration(self, ch: str, record: dict) -> float:
        """计算单通道恶化度: 0=完全健康, 1=严重恶化。"""
        z = self.baseline.z_scores(record)[ch]
        if ch in self.ONE_SIDED:
            # 单边: z>0.5σ 起算(去死区), 各通道满量程跨度经真值校准
            span = self.SPAN_ONE_SIDED[ch]
            return float(np.clip((z - self.Z0_ONE_SIDED) / span, 0.0, 1.0))
        # 双边通道(电压/转速): |z| 偏离即恶化
        return float(np.clip((abs(z) - self.Z0_TWO_SIDED) / self.SPAN_TWO_SIDED,
                             0.0, 1.0))

    @staticmethod
    def calibrate(raw_score: float) -> float:
        """保序回归分段线性校准: 原始分 -> 校准分(≈100×真值健康度)。"""
        knots = HealthScorer.SCORE_CALIBRATION
        return float(np.interp(raw_score, [k[0] for k in knots],
                               [k[1] for k in knots]))

    def score(self, record: dict) -> float:
        """加权健康评分(0~100, 保留 1 位小数)。"""
        det = sum(CHANNEL_WEIGHTS[ch] * self.channel_deterioration(ch, record)
                  for ch in CHANNELS)
        return round(self.calibrate(100.0 * (1.0 - det)), 1)


# ------------------------------------------------------------------------------
# RUL 剩余寿命预测器: 趋势外推 + 置信区间
# ------------------------------------------------------------------------------
class RULPredictor:
    """基于健康评分退化趋势的剩余寿命预测。

    方法:
        1. 取最近 RUL_WINDOW 条健康评分序列;
        2. 最小二乘拟合一次退化趋势(每日下降速率), 可选检测加速退化;
        3. 外推至失效阈值 FAILURE_THRESHOLD 得 RUL(小时);
        4. 用拟合残差的标准差构造 80% 置信区间: 残差属"评分点"量纲,
           先按线性外推预测方差放大
           margin_score = 1.2816·σ·sqrt(1 + 1/n + (k-x̄)²/Sxx)
           (k 为外推步数, 1.2816 为标准正态 80% 分位点), 再除以拟合
           斜率绝对值 |b| 换算为时间量纲(小时)。
    """

    def predict(self, health_series: list) -> dict:
        """对健康评分序列(按时间升序, 0~100)预测 RUL。

        输入契约: 序列必须按"1 条 = 1 tick(10 分钟)"等间隔采样——调用方
        (看板流水线 / 评估脚本)均逐 tick 喂入, 见模块 TICK_HOURS 处的
        喂入口径契约说明。

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
            # 评分未下降: 趋势不足, 不强行外推(返回 None 并标注), 避免虚假承诺
            return {"rul_hours": None, "rul_ci_low": None, "rul_ci_high": None,
                    "decline_per_day": round(decline_per_day, 2), "trend": trend,
                    "method": "trend_extrapolation"}
        ticks_to_fail = (current - FAILURE_THRESHOLD) / (-b)
        rul_hours = ticks_to_fail * TICK_HOURS
        # 80% 置信区间(2026-08 修复审查报告 07-P1-1 的量纲错误):
        # 残差 sigma 是"评分点"量纲, 须先按线性外推预测方差放大到失效
        # 时刻(k = ticks_to_fail), 再除以拟合斜率绝对值 |b|(评分点/tick)
        # 换算为 tick, 最后乘 TICK_HOURS 得小时。v1.0 直接 margin×TICK_HOURS
        # 得到"评分点×小时/tick", |b| 越小(慢退化, 恰是 RUL 最有价值的
        # 早期)CI 被压得越窄。
        x_bar = (n - 1) / 2.0
        sxx = float(np.sum((x - x_bar) ** 2))
        inflation = math.sqrt(1.0 + 1.0 / n
                              + (max(ticks_to_fail, 0.0) - x_bar) ** 2
                              / max(sxx, 1e-9))
        margin_score = 1.2816 * max(sigma, 0.5) * inflation
        margin_hours = margin_score / max(abs(b), 1e-9) * TICK_HOURS
        # 上限截断: RUL 与置信区间一致截断到运维规划视野内
        ci_low = max(0.0, rul_hours - margin_hours)
        ci_high = min(RUL_MAX_HOURS, rul_hours + margin_hours)
        rul_hours = min(RUL_MAX_HOURS, rul_hours)
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
        # 规则回退: 直接以健康评分阈值划分。校准评分 ≈ 100×真值健康度,
        # 阈值 80/60 与模拟器标签边界(normal>0.8, warning>0.6)对齐
        if health_score > 80.0:
            return "normal"
        return "warning" if health_score > 60.0 else "fault"


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
