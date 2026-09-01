# -*- coding: utf-8 -*-
"""
================================================================================
 告警引擎 (Alert Engine)
================================================================================
模块职责:
    1. 规则告警: 设备健康评分跌破分级阈值(提醒/严重/紧急)时自动产生告警;
    2. 预测告警: AI 引擎预测的剩余寿命 RUL 低于设定窗口(如 72 小时)时,
       提前发出"潜在故障预警", 这正是预测性维护"事前预防"的核心动作;
    3. 工单生成: 告警触发后自动创建维修工单, 通过故障分类器匹配最可能
       故障并关联知识库中的处理方案/备件清单, 实现告警->诊断->工单的
       闭环自动化;
    4. 告警抑制: 同一设备同一级别告警在抑制窗口内不重复产生, 防止告警
       风暴; 告警升级: 更高级别到来时关闭旧告警并继承上下文。

运行示例:
    python alert_engine.py --selftest     # 内置场景自检
================================================================================
"""

import argparse
import json
import os
import sys
import threading
from datetime import datetime

# 引入同项目的故障分类器(知识库), 用于告警自动关联处理方案
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ai-prediction"))
from fault_classifier import FaultClassifier            # noqa: E402

ALERT_DIR = os.path.join(PROJECT_ROOT, "data", "alerts")
ALERTS_PATH = os.path.join(ALERT_DIR, "alerts.json")
WORKORDERS_PATH = os.path.join(ALERT_DIR, "workorders.json")

# ------------------------------------------------------------------------------
# 分级阈值与等级定义
# ------------------------------------------------------------------------------
# 健康评分阈值: 75 以下提醒, 55 以下严重, 35 以下紧急
HEALTH_THRESHOLDS = [
    (35.0, "emergency"),      # 紧急: 立即停机处置
    (55.0, "critical"),       # 严重: 24 小时内处置
    (75.0, "warning"),        # 提醒: 安排巡检
]
SEVERITY_NAMES = {"warning": "提醒", "critical": "严重", "emergency": "紧急",
                  "predictive": "预测预警"}
# RUL 预警窗口(小时): 预计剩余寿命低于该值即发预测预警
RUL_ALERT_HOURS = 72.0
# 同级告警抑制窗口(秒, 演示用短窗口; 生产建议 6~12 小时)
SUPPRESS_SECONDS = 60.0

# 与数据模拟器一致的设备类型映射(告警文案使用)
DEVICE_TYPES = {"MOTOR-001": "motor", "FAN-001": "fan", "GEARBOX-001": "gearbox"}
DEVICE_NAMES = {"MOTOR-001": "主电机", "FAN-001": "离心风机", "GEARBOX-001": "齿轮传动箱"}


class AlertEngine:
    """告警与工单管理引擎。

    数据模型:
        alert    = {id, device_id, severity, title, detail, ts, status,
                   health_score, rul_hours, suggestions[]}
        workorder= {id, alert_id, device_id, fault_code, fault_name,
                    confidence, actions[], spare_parts[], status, created}
    """

    def __init__(self, alerts_path: str = ALERTS_PATH,
                 workorders_path: str = WORKORDERS_PATH):
        self.alerts_path = alerts_path
        self.workorders_path = workorders_path
        self.classifier = FaultClassifier()
        self.alerts = []          # 活动告警列表
        self.workorders = []      # 工单列表
        self._counter = 0
        # 跨线程互斥锁(2026-08 修复审查报告 07-P1-4): 流水线线程调用
        # evaluate(), Flask 线程并发调用 close_workorder()/查询接口/
        # 复位路径, 无锁时 json.dump 遍历列表期间被并发 append 会抛
        # RuntimeError 或写出交错文件。用 RLock 允许 evaluate() 持锁
        # 期间内部再调用 _save()。
        self.lock = threading.RLock()
        self._load()

    # ---------------------------- 持久化 ----------------------------
    def _load(self):
        """启动时恢复历史告警与工单(演示环境自动清理 7 天前记录)。"""
        for path, attr in ((self.alerts_path, "alerts"),
                           (self.workorders_path, "workorders")):
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        setattr(self, attr, json.load(f))
                except (json.JSONDecodeError, OSError):
                    setattr(self, attr, [])
        # 计数器取已加载记录的最大数字序号: _save 会把列表截断到最近 500 条,
        # 若按条目数恢复计数器, 重启后新 ID 可能与被裁剪掉编号段之后的历史
        # 记录重复, 破坏工单-告警关联与按 ID 关闭工单的准确性。
        self._counter = self._max_existing_seq()

    def _max_existing_seq(self) -> int:
        """返回已加载告警/工单中最大的自增序号(无记录时为 0)。"""
        mx = 0
        for items in (self.alerts, self.workorders):
            for item in items:
                try:
                    mx = max(mx, int(str(item.get("id", "")).rsplit("-", 1)[-1]))
                except (ValueError, AttributeError, IndexError):
                    continue
        return mx

    @staticmethod
    def _atomic_dump(path: str, data) -> None:
        """原子写 JSON: 先写同目录临时文件, 再 os.replace 原子替换。

        直接 open(path, "w") 在写入途中被并发读取/进程中断会留下半截
        文件, 下次启动 json.load 失败; os.replace 在同一文件系统上原子生效。
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)

    def _save(self):
        """告警与工单落盘, 供看板读取与断电恢复(持锁 + 原子替换)。"""
        with self.lock:
            # 写文件前截断到最近 500 条, 防止演示环境磁盘膨胀
            self._atomic_dump(self.alerts_path, self.alerts[-500:])
            self._atomic_dump(self.workorders_path, self.workorders[-500:])

    def _next_id(self, prefix: str) -> str:
        """生成自增编号: ALT-0001 / WO-0001 ..."""
        self._counter += 1
        return "%s-%04d" % (prefix, self._counter)

    # ---------------------------- 告警评估 ----------------------------
    def evaluate(self, prediction: dict, z_scores: dict = None) -> list:
        """对单台设备的一次预测结果做告警评估(引擎主入口)。

        Args:
            prediction: AI 引擎输出 {device_id, health_score, rul{...}, ...}
            z_scores:   通道 z 分数(供故障分类器归因), 缺省用空字典

        Returns:
            本轮新生成的告警列表(可能为空)。
        """
        did = prediction["device_id"]
        health = prediction.get("health_score", 100.0)
        rul = prediction.get("rul") or {}
        rul_hours = rul.get("rul_hours")
        z_scores = z_scores or prediction.get("features") or {}
        now = datetime.now().isoformat(timespec="seconds")
        fired = []

        # 全程持锁: 与 Flask 线程的 close_workorder()/查询/复位互斥
        with self.lock:
            # ---- 规则一: 健康评分分级告警 ----
            for threshold, severity in HEALTH_THRESHOLDS:
                if health < threshold:
                    alert = self._fire_health_alert(did, severity, threshold,
                                                    health, rul_hours, z_scores, now)
                    if alert:
                        fired.append(alert)
                    break                   # 只取满足的最高级别

            # ---- 规则二: RUL 预测预警(即便健康评分尚可) ----
            if rul_hours is not None and rul_hours < RUL_ALERT_HOURS and health >= 75.0:
                alert = self._fire_predictive_alert(did, rul, health, z_scores, now)
                if alert:
                    fired.append(alert)

            if fired:
                self._save()
        return fired

    def _suppressed(self, device_id: str, severity: str) -> bool:
        """同设备同级告警在抑制窗口内不重复。"""
        now = datetime.now()
        for a in self.alerts:
            if (a["device_id"] == device_id and a["severity"] == severity
                    and a["status"] == "active"):
                prev = datetime.fromisoformat(a["ts"])
                if (now - prev).total_seconds() < SUPPRESS_SECONDS:
                    return True
        return False

    def _close_lower_levels(self, device_id: str, severity: str):
        """高级别告警出现时, 关闭同设备较低级别的活动告警(升级)。"""
        order = {"warning": 0, "predictive": 1, "critical": 2, "emergency": 3}
        for a in self.alerts:
            if (a["device_id"] == device_id and a["status"] == "active"
                    and order.get(a["severity"], 0) < order.get(severity, 0)):
                a["status"] = "superseded"
                a["closed_ts"] = datetime.now().isoformat(timespec="seconds")

    def _diagnose(self, device_id: str, z_scores: dict, health: float) -> dict:
        """调用故障分类器完成自动归因, 返回匹配结果。"""
        device_type = DEVICE_TYPES.get(device_id, "motor")
        return self.classifier.classify(z_scores, device_type, health_score=health)

    # ---------------------------- 告警生成 ----------------------------
    def _fire_health_alert(self, did, severity, threshold, health,
                           rul_hours, z_scores, now) -> dict:
        """生成健康评分告警, 并联动创建维修工单。"""
        if self._suppressed(did, severity):
            return None
        self._close_lower_levels(did, severity)
        diagnosis = self._diagnose(did, z_scores, health)
        top = diagnosis["matched"][0] if diagnosis["matched"] else None
        title = "%s %s 健康评分 %.1f 低于阈值 %.0f" % (
            DEVICE_NAMES.get(did, did), SEVERITY_NAMES[severity], health, threshold)
        alert = {
            "id": self._next_id("ALT"),
            "device_id": did,
            "severity": severity,
            "severity_name": SEVERITY_NAMES[severity],
            "title": title,
            "detail": diagnosis["summary"],
            "ts": now,
            "status": "active",
            "health_score": health,
            "rul_hours": rul_hours,
            "fault_code": top["code"] if top else None,
        }
        self.alerts.append(alert)
        # 紧急/严重级别自动生成工单; 提醒级别仅建议巡检
        if severity in ("critical", "emergency"):
            wo = self._create_workorder(alert, diagnosis)
            alert["workorder_id"] = wo["id"]
            print("[告警引擎] %s -> 已自动生成工单 %s" % (alert["id"], wo["id"]))
        return alert

    def _fire_predictive_alert(self, did, rul, health, z_scores, now) -> dict:
        """生成 RUL 预测预警(潜在故障提前预告)。"""
        if self._suppressed(did, "predictive"):
            return None
        diagnosis = self._diagnose(did, z_scores, health)
        alert = {
            "id": self._next_id("ALT"),
            "device_id": did,
            "severity": "predictive",
            "severity_name": SEVERITY_NAMES["predictive"],
            "title": ("%s 预测 %.1f 小时后触及失效阈值(80%%CI: %.1f~%.1f)"
                      % (DEVICE_NAMES.get(did, did), rul["rul_hours"],
                         rul.get("rul_ci_low", 0), rul.get("rul_ci_high", 0))),
            "detail": diagnosis["summary"],
            "ts": now,
            "status": "active",
            "health_score": health,
            "rul_hours": rul["rul_hours"],
            "fault_code": (diagnosis["matched"][0]["code"]
                           if diagnosis["matched"] else None),
        }
        self.alerts.append(alert)
        return alert

    # ---------------------------- 工单生成 ----------------------------
    def _create_workorder(self, alert: dict, diagnosis: dict) -> dict:
        """根据诊断结果创建维修工单(关联故障库方案与备件)。"""
        matched = diagnosis["matched"]
        top = matched[0] if matched else None
        workorder = {
            "id": self._next_id("WO"),
            "alert_id": alert["id"],
            "device_id": alert["device_id"],
            "device_name": DEVICE_NAMES.get(alert["device_id"], alert["device_id"]),
            "fault_code": top["code"] if top else None,
            "fault_name": top["name"] if top else "待诊断异常",
            "confidence": top["confidence"] if top else 0.0,
            "severity": alert["severity"],
            "actions": self.classifier.recommend_actions(matched),
            "spare_parts": top["spare_parts"] if top else [],
            "deadline_hours": top["risk_of_failure_hours"] if top else 48,
            "status": "open",                 # open / in_progress / closed
            "created": datetime.now().isoformat(timespec="seconds"),
        }
        self.workorders.append(workorder)
        return workorder

    # ---------------------------- 查询接口 ----------------------------
    def active_alerts(self, device_id: str = None) -> list:
        """查询活动告警(可按设备过滤), 按时间倒序(持锁取快照)。"""
        with self.lock:
            items = [a for a in self.alerts if a["status"] == "active"]
        if device_id:
            items = [a for a in items if a["device_id"] == device_id]
        return sorted(items, key=lambda a: a["ts"], reverse=True)

    def open_workorders(self, device_id: str = None) -> list:
        """查询未关闭工单(可按设备过滤, 持锁取快照)。"""
        with self.lock:
            items = [w for w in self.workorders if w["status"] != "closed"]
        if device_id:
            items = [w for w in items if w["device_id"] == device_id]
        return items

    def close_workorder(self, workorder_id: str) -> bool:
        """关闭工单(维修完成), 同时关闭其来源告警。"""
        with self.lock:
            for w in self.workorders:
                if w["id"] == workorder_id:
                    w["status"] = "closed"
                    w["closed"] = datetime.now().isoformat(timespec="seconds")
                    for a in self.alerts:
                        if a.get("workorder_id") == workorder_id:
                            a["status"] = "resolved"
                    self._save()
                    return True
            return False

    def resolve_all(self) -> None:
        """把全部活动告警置为 resolved 并落盘(演示复位时由看板调用)。

        v1.0 中看板复位直接跨线程改 self.alerts 并调 _save(), 与流水线
        线程的 evaluate() 竞态; 现在收敛为本方法统一持锁处理。
        """
        with self.lock:
            for a in self.alerts:
                if a["status"] == "active":
                    a["status"] = "resolved"
            self._save()


# ------------------------------------------------------------------------------
# 内置自检: 模拟三档健康评分与 RUL 场景, 验证告警与工单联动
# ------------------------------------------------------------------------------
def run_selftest() -> bool:
    """构造递进恶化的预测结果序列, 验证告警分级/抑制/工单生成。

    自检全程使用系统临时目录, 不读写真实告警库 data/alerts/,
    避免测试告警/工单污染生产状态文件。
    """
    import tempfile
    with tempfile.TemporaryDirectory(prefix="alert_selftest_") as tmp:
        engine = AlertEngine(alerts_path=os.path.join(tmp, "alerts.json"),
                             workorders_path=os.path.join(tmp, "workorders.json"))
        return _run_selftest_scenarios(engine)


def _run_selftest_scenarios(engine: AlertEngine) -> bool:
    """在给定引擎实例上执行递进恶化场景自检(由 run_selftest 调用)。"""
    device = "MOTOR-001"
    scenarios = [
        # (健康评分, 振动z, 温度z, 电流z, 说明)
        (78.0, 0.5, 0.3, 0.2, "健康: 无告警"),
        (70.0, 1.8, 1.0, 0.5, "提醒级: 健康跌破75"),
        (52.0, 2.2, 1.4, 0.8, "严重级: 跌破55, 应生成工单"),
        (30.0, 4.5, 2.8, 1.5, "紧急级: 跌破35, 升级并再生成工单"),
    ]
    print("[告警引擎] 自检开始 (设备: %s)\n" % device)
    ok = True
    for health, zv, zt, zc, desc in scenarios:
        prediction = {
            "device_id": device, "health_score": health,
            "rul": {"rul_hours": 40.0 if health < 55 else 300.0,
                    "rul_ci_low": 30.0, "rul_ci_high": 60.0},
            "features": {"vibration": zv, "temperature": zt, "current": zc,
                         "voltage": 0.1, "rpm": 0.0},
        }
        fired = engine.evaluate(prediction)
        alert_titles = [a["title"] for a in fired]
        print("场景[%s] -> 触发 %d 条告警" % (desc, len(fired)))
        for t in alert_titles:
            print("   - %s" % t)
        if health == 78.0 and fired:
            ok = False
        if health == 52.0 and not any(a["severity"] == "critical" for a in fired):
            ok = False
        if health == 30.0 and not any(a["severity"] == "emergency" for a in fired):
            ok = False
    print("\n工单列表:")
    for w in engine.open_workorders():
        print("  %s %s 故障=%s 置信度=%.0f%% 措施数=%d"
              % (w["id"], w["device_id"], w["fault_name"],
                 w["confidence"] * 100, len(w["actions"])))
    ok = ok and len(engine.open_workorders()) >= 2
    print("\n自检结果: %s" % ("全部通过" if ok else "存在失败"))
    return ok


def main():
    parser = argparse.ArgumentParser(description="数字孪生平台 - 告警引擎")
    parser.add_argument("--selftest", action="store_true", help="运行内置场景自检")
    args = parser.parse_args()
    # 自检失败以非零退出码上报, 便于脚本/CI 感知
    sys.exit(0 if run_selftest() else 1)


if __name__ == "__main__":
    main()
