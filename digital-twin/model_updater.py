# -*- coding: utf-8 -*-
"""
================================================================================
 数字孪生模型更新器 (Digital Twin Model Updater)
================================================================================
模块职责:
    维护每台物理设备在信息空间中的"数字镜像"(Twin Model), 实现:
      1. 物理 -> 数字 (P2D): 消费实时传感器数据, 校准数字模型的状态参数
         (基线漂移补偿、负载因子估计、模型增益自适应), 使数字模型逐步
         逼近真实设备的当前行为;
      2. 数字 -> 物理 (D2P): 当数字模型检测到实体与模型的偏差超限时,
         生成反向控制建议(降载运行、提前巡检、启停建议), 模拟"孪生体
         反哺物理世界"的闭环;
      3. 偏差记录: 持续记录每个传感器通道的残差(actual - expected)、
         偏差百分比、EWMA 平滑偏差与总体同步度, 为 AI 预测引擎提供
         "模型-实体一致性"特征。

同步机制说明:
    数字模型对每个通道维护一个"校准基线"(初始取设备铭牌参数)。每条
    实时数据到来时:
        expected = calibrated_baseline * (1 + load_factor * 响应系数)
        residual = actual - expected
        calibrated_baseline += learning_rate * residual   (EWMA 在线校准)
    校准后的模型即设备当前状态的"数字化身"; 残差序列则反映实体中模型
    尚未解释的异常成分 —— 正是早期故障特征的来源。

运行示例:
    python model_updater.py --once      # 单步演示(内置模拟数据)
    python model_updater.py --follow    # 跟随 data/stream/ 数据流持续同步
================================================================================
"""

import argparse
import copy
import json
import math
import os
import time
from datetime import datetime

# 项目根目录与运行时状态目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TWIN_DIR = os.path.join(PROJECT_ROOT, "data", "twin")
TWIN_STATE_PATH = os.path.join(TWIN_DIR, "twin_state.json")
STREAM_DIR = os.path.join(PROJECT_ROOT, "data", "stream")

# 与数据模拟器一致的设备基线(独立维护, 避免跨目录模块依赖)
DEVICE_PROFILES = {
    "MOTOR-001": {"name": "主电机", "type": "motor",
                  "nominal": {"vibration": 1.6, "temperature": 55.0,
                              "current": 82.0, "voltage": 380.0}},
    "FAN-001": {"name": "离心风机", "type": "fan",
                "nominal": {"vibration": 2.4, "temperature": 48.0,
                            "current": 56.0, "voltage": 380.0}},
    "GEARBOX-001": {"name": "齿轮传动箱", "type": "gearbox",
                    "nominal": {"vibration": 2.0, "temperature": 62.0,
                                "current": 74.0, "voltage": 380.0}},
}

# 各通道的健康响应系数: 用于由"同步度"反推数字模型视角下的健康估计
CHANNEL_SENSITIVITY = {"vibration": 3.2, "temperature": 30.0,
                       "current": 0.22, "voltage": 6.0}


class TwinModel:
    """单台设备的数字孪生模型。

    属性:
        calibrated   校准后的各通道基线(数字模型的核心状态参数)
        load_factor  估计的负载因子(0.85~1.15)
        residuals    最近 N 条残差记录 [{ts, channel: residual}]
        ewma         各通道残差的指数加权滑动平均(长期漂移指示)
        deviations   偏差统计历史 [{ts, mape, sync_score, health_est}]
        suggestions  数字->物理 反向控制建议列表
    """

    HISTORY_LIMIT = 300        # 内存中保留的最大历史条数
    LEARNING_RATE = 0.05       # 在线校准学习率(越小越保守)
    EWMA_ALPHA = 0.20          # 残差平滑系数
    DEVIATION_ALERT = 0.15        # 单通道偏差率超过 15% 触发反向建议

    def __init__(self, device_id: str, profile: dict):
        self.device_id = device_id
        self.name = profile["name"]
        self.device_type = profile["type"]
        # 深拷贝铭牌基线作为校准起点(数字模型的"出厂状态")
        self.calibrated = copy.deepcopy(profile["nominal"])
        self.nominal = copy.deepcopy(profile["nominal"])
        self.load_factor = 1.0
        self.last_update = None
        self.residuals = []        # 最近残差记录(按时间序列)
        self.ewma = {ch: 0.0 for ch in self.calibrated}
        self.deviations = []       # 偏差统计历史
        self.suggestions = []      # 反向控制建议

    # ------------------------------------------------------------------
    # 数字模型正向计算: 给定当前模型状态, 推演传感器"应有"的读数
    # ------------------------------------------------------------------
    def expected_readings(self) -> dict:
        """根据校准基线与负载因子, 推演当前工况下的期望传感器值。

        物理意义:
            振动/电流 随负载近线性放大; 温度随负载摩擦生热二次上升;
            电压在重载时略有压降(母线阻抗)。
        """
        lf = self.load_factor
        return {
            "vibration": self.calibrated["vibration"] * lf,
            "temperature": self.calibrated["temperature"] + 12.0 * (lf - 1.0) ** 2 / 0.15,
            "current": self.calibrated["current"] * lf,
            "voltage": self.calibrated["voltage"] - 3.0 * (lf - 1.0) / 0.3,
        }

    # ------------------------------------------------------------------
    # 物理 -> 数字: 用实测数据更新数字模型
    # ------------------------------------------------------------------
    def update(self, record: dict) -> dict:
        """消费一条实时采样记录, 完成一次孪生同步。

        步骤:
            1) 用泰勒展开思想从电流通道估计负载因子;
            2) 计算各通道期望值与残差;
            3) EWMA 在线校准基线(缓慢吸收系统性漂移);
            4) 更新偏差统计与总体同步度;
            5) 偏差超限时生成数字->物理的反向控制建议。

        Returns:
            同步结果摘要 dict(含各通道偏差与同步度评分)。
        """
        ts = record.get("timestamp", datetime.now().isoformat())
        self.last_update = ts

        # ---- 步骤 1: 负载因子估计(电流对负载最敏感) ----
        cur_expected_idle = self.calibrated["current"]
        lf_est = record.get("current", cur_expected_idle) / max(1.0, cur_expected_idle)
        self.load_factor += 0.10 * (lf_est - self.load_factor)   # 平滑跟踪
        self.load_factor = max(0.80, min(1.20, self.load_factor))

        # ---- 步骤 2: 残差计算 ----
        expected = self.expected_readings()
        residual_detail = {}
        for channel, exp in expected.items():
            actual = record.get(channel)
            if actual is None:
                continue
            residual = actual - exp
            pct = residual / exp if abs(exp) > 1e-6 else 0.0
            residual_detail[channel] = {"actual": round(actual, 3),
                                        "expected": round(exp, 3),
                                        "residual": round(residual, 3),
                                        "pct": round(pct, 4)}
            # EWMA 平滑残差(长期漂移指示)
            self.ewma[channel] = self.EWMA_ALPHA * residual + (1 - self.EWMA_ALPHA) * self.ewma[channel]

        # ---- 步骤 3: 在线校准(只吸收方向一致且幅度有限的漂移) ----
        for channel, detail in residual_detail.items():
            drift = self.LEARNING_RATE * detail["residual"]
            # 限制单次校准幅度不超过基线的 2%, 防止异常点污染模型
            limit = 0.02 * self.calibrated[channel]
            drift = max(-limit, min(limit, drift))
            self.calibrated[channel] = round(self.calibrated[channel] + drift, 4)

        # ---- 步骤 4: 偏差统计与同步度 ----
        summary = self._compute_deviation(ts, residual_detail)

        # ---- 步骤 5: 反向控制建议(数字 -> 物理) ----
        self._maybe_suggest(ts, residual_detail, summary)

        # 记录历史(截断防膨胀)
        self.residuals.append({"ts": ts, "detail": residual_detail})
        if len(self.residuals) > self.HISTORY_LIMIT:
            self.residuals = self.residuals[-self.HISTORY_LIMIT:]
        return summary

    def _compute_deviation(self, ts: str, residual_detail: dict) -> dict:
        """汇总各通道偏差, 计算平均绝对偏差率(MAPE)与同步度评分。

        同步度 = 100 * exp(-8 * MAPE): MAPE=0 完全同步, MAPE 越大指数衰减。
        数字模型视角下的健康估计 health_est = 同步度/100 与"校准基线相对
        铭牌的累计漂移"综合而成, 供预测引擎交叉验证。
        """
        pcts = [abs(d["pct"]) for d in residual_detail.values()]
        mape = sum(pcts) / len(pcts) if pcts else 0.0
        sync_score = 100.0 * math.exp(-8.0 * mape)
        # 基线累计漂移: 校准基线相对铭牌偏离越大, 说明设备性能劣化越深
        drift_terms = []
        for ch, sens in CHANNEL_SENSITIVITY.items():
            nom = self.nominal[ch]
            if ch == "temperature":
                drift = (self.calibrated[ch] - nom) / sens          # 温度: 除以温升幅度
            else:
                drift = (self.calibrated[ch] - nom) / nom            # 其余: 相对偏离
            drift_terms.append(max(0.0, drift))
        avg_drift = sum(drift_terms) / len(drift_terms)
        health_est = max(0.0, min(1.0, 1.0 - 1.6 * avg_drift)) * (sync_score / 100.0)
        summary = {
            "device_id": self.device_id,
            "ts": ts,
            "mape": round(mape, 4),
            "sync_score": round(sync_score, 1),
            "health_est": round(health_est, 4),
            "load_factor": round(self.load_factor, 3),
            "channels": residual_detail,
        }
        self.deviations.append(summary)
        if len(self.deviations) > self.HISTORY_LIMIT:
            self.deviations = self.deviations[-self.HISTORY_LIMIT:]
        return summary

    def _maybe_suggest(self, ts: str, residual_detail: dict, summary: dict):
        """当实体与数字模型偏差持续超限时, 生成数字->物理的反向控制建议。"""
        for channel, detail in residual_detail.items():
            if abs(detail["pct"]) < self.DEVIATION_ALERT:
                continue
            direction = "偏高" if detail["pct"] > 0 else "偏低"
            # 针对不同通道给出有物理依据的处置建议
            advice = {
                "vibration": "振动%s: 建议安排轴承/联轴器检查, 必要时降转速 10%% 运行" % direction,
                "temperature": "温度%s: 检查冷却与润滑回路, 建议降载 15%% 观察温升趋势" % direction,
                "current": "电流%s: 排查机械卡滞与绝缘状态, 建议错峰降载" % direction,
                "voltage": "电压%s: 检查供电回路与接点, 防止欠压过流" % direction,
            }.get(channel, "通道 %s 偏差超限, 建议现场核查" % channel)
            self.suggestions.append({
                "ts": ts, "device_id": self.device_id, "channel": channel,
                "pct": round(detail["pct"], 3), "advice": advice,
                "sync_score": summary["sync_score"],
            })
        # 建议列表只保留最近 50 条
        if len(self.suggestions) > 50:
            self.suggestions = self.suggestions[-50:]

    def reset(self):
        """把孪生体复位到"出厂状态"(演示复位时调用)。

        复位内容: 校准基线回到铭牌值、负载因子回 1.0、清空 EWMA 残差、
        偏差历史与反向建议。若不复位, 演示复位后孪生仍带着上一轮寿命的
        累计漂移, 同步度与模型健康估计会停留在低位, 与"全新设备"矛盾。
        """
        self.calibrated = copy.deepcopy(self.nominal)
        self.load_factor = 1.0
        self.last_update = None
        self.residuals = []
        self.ewma = {ch: 0.0 for ch in self.calibrated}
        self.deviations = []
        self.suggestions = []

    # ------------------------------------------------------------------
    # 导出: 供 AI 预测引擎与看板使用的数字孪生快照
    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        """导出数字孪生当前完整状态(可序列化为 JSON)。"""
        return {
            "device_id": self.device_id,
            "name": self.name,
            "device_type": self.device_type,
            "last_update": self.last_update,
            "calibrated_baseline": self.calibrated,
            "nominal_baseline": self.nominal,
            "load_factor": round(self.load_factor, 3),
            "ewma_residual": {k: round(v, 4) for k, v in self.ewma.items()},
            "sync_score": self.deviations[-1]["sync_score"] if self.deviations else 100.0,
            "health_est": self.deviations[-1]["health_est"] if self.deviations else 1.0,
            "recent_deviations": self.deviations[-30:],
            "suggestions": self.suggestions[-10:],
        }

    def deviation_report(self) -> str:
        """生成人可读的偏差报告文本(用于控制台/导出)。"""
        lines = ["[%s] %s(%s) 数字孪生偏差报告" % (self.last_update, self.name, self.device_id)]
        lines.append("  负载因子=%.3f  同步度=%.1f  模型健康估计=%.2f"
                     % (self.load_factor,
                        self.deviations[-1]["sync_score"] if self.deviations else 100.0,
                        self.deviations[-1]["health_est"] if self.deviations else 1.0))
        for ch, d in (self.deviations[-1]["channels"] if self.deviations else {}).items():
            lines.append("  %-10s 实测=%8.3f  模型=%8.3f  残差=%+8.3f (%+.1f%%)"
                         % (ch, d["actual"], d["expected"], d["residual"], d["pct"] * 100))
        return "\n".join(lines)


class TwinUpdater:
    """管理全部设备数字孪生的编排器: 批量更新 / 持久化 / 跟随数据流。"""

    def __init__(self, state_path: str = TWIN_STATE_PATH):
        self.state_path = state_path
        self.twins = {}
        self._load()

    # ---------------------------- 持久化 ----------------------------
    def _load(self):
        """从磁盘恢复数字孪生状态(不存在则按铭牌参数初始化全新孪生)。"""
        state = {}
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
            except (json.JSONDecodeError, OSError):
                state = {}
        for device_id, profile in DEVICE_PROFILES.items():
            if device_id in state:
                twin = TwinModel(device_id, profile)
                saved = state[device_id]
                twin.calibrated = saved.get("calibrated_baseline", twin.calibrated)
                twin.load_factor = saved.get("load_factor", 1.0)
                twin.last_update = saved.get("last_update")
                twin.ewma = saved.get("ewma_residual", twin.ewma)
                twin.deviations = saved.get("recent_deviations", [])
                twin.suggestions = saved.get("suggestions", [])
                self.twins[device_id] = twin
            else:
                self.twins[device_id] = TwinModel(device_id, profile)

    def save(self):
        """将全部孪生状态落盘(断电重启后数字模型可无缝恢复)。"""
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        state = {did: twin.snapshot() for did, twin in self.twins.items()}
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    # ---------------------------- 批量同步 ----------------------------
    def update_all(self, records: list) -> list:
        """批量消费多条采样记录, 返回同步结果摘要列表。"""
        summaries = []
        for rec in records:
            did = rec.get("device_id")
            if did in self.twins:
                summaries.append(self.twins[did].update(rec))
        if summaries:
            self.save()
        return summaries

    def snapshots(self) -> dict:
        """导出全部孪生快照(看板 /api/twin 接口的数据源)。"""
        return {did: twin.snapshot() for did, twin in self.twins.items()}

    def reset_all(self):
        """全部孪生复位到出厂状态并落盘(演示复位时由看板调用)。"""
        for twin in self.twins.values():
            twin.reset()
        self.save()


def read_latest_records(stream_dir: str = STREAM_DIR) -> list:
    """读取数据流中每台设备最新的一条记录(跟随模式的数据输入)。

    数据流由 data_simulator.py 以 JSONL 追加方式写入; 本函数读取每个
    文件的最后一行, 模拟"增量订阅"效果。
    """
    records = []
    if not os.path.isdir(stream_dir):
        return records
    for filename in os.listdir(stream_dir):
        if not filename.endswith(".jsonl"):
            continue
        path = os.path.join(stream_dir, filename)
        try:
            with open(path, "rb") as f:
                # 高效读取最后一行: 移到文件尾部向前找换行符
                f.seek(0, os.SEEK_END)
                end = f.tell()
                tail = b""
                for pos in range(end - 1, max(-1, end - 4096), -1):
                    f.seek(pos)
                    ch = f.read(1)
                    tail = ch + tail
                    if ch == b"\n" and pos < end - 1:
                        break
                line = [l for l in tail.decode("utf-8", "ignore").splitlines() if l.strip()]
                if line:
                    records.append(json.loads(line[-1]))
        except (OSError, json.JSONDecodeError):
            continue
    return records


def demo_once():
    """单步演示: 用内置模拟数据驱动一轮孪生同步, 打印偏差报告。"""
    sys_path_hack = os.path.join(PROJECT_ROOT, "data-ingestion")
    import sys
    sys.path.insert(0, sys_path_hack)
    from data_simulator import DataSimulator          # 复用数据模拟器

    sim = DataSimulator(seed=99)
    updater = TwinUpdater()
    print("[数字孪生] 演示: 快进 900 个循环的设备生命周期 ...")
    for round_no in range(900):
        records = sim.step_all()
        for rec in records:
            updater.twins[rec["device_id"]].update(rec)
        # 在关键阶段打印偏差报告
        if round_no in (0, 300, 600, 899):
            print("\n===== 同步轮次 %d =====" % round_no)
            for twin in updater.twins.values():
                print(twin.deviation_report())
    updater.save()
    print("\n[数字孪生] 状态已持久化 -> %s" % updater.state_path)


def demo_follow(interval: float = 2.0):
    """跟随模式: 持续读取 data/stream/ 最新数据并同步孪生模型。"""
    updater = TwinUpdater()
    print("[数字孪生] 跟随数据流: %s (每 %.1f 秒同步一次, Ctrl+C 退出)" % (STREAM_DIR, interval))
    seen_ts = {}
    try:
        while True:
            for rec in read_latest_records():
                did = rec["device_id"]
                if seen_ts.get(did) == rec.get("timestamp"):
                    continue                     # 跳过未更新的记录
                seen_ts[did] = rec.get("timestamp")
                summary = updater.twins[did].update(rec)
                print("[%s] %s 同步度=%.1f 模型健康估计=%.3f"
                      % (summary["ts"], did, summary["sync_score"], summary["health_est"]))
            updater.save()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[数字孪生] 已停止。")


def main():
    parser = argparse.ArgumentParser(description="数字孪生平台 - 模型更新器")
    parser.add_argument("--once", action="store_true", help="单步演示(内置模拟数据)")
    parser.add_argument("--follow", action="store_true", help="跟随 data/stream/ 数据流")
    parser.add_argument("--interval", type=float, default=2.0, help="跟随模式同步周期(秒)")
    args = parser.parse_args()
    if args.follow:
        demo_follow(args.interval)
    else:
        demo_once()


if __name__ == "__main__":
    main()
