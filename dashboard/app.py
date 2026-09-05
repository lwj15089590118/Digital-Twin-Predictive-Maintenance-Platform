# -*- coding: utf-8 -*-
"""
================================================================================
 Flask 综合看板 (Dashboard)
================================================================================
模块职责:
    1. 后台流水线: 在独立线程中以演示节奏驱动
       "数据模拟器 -> 数字孪生更新 -> AI 预测 -> 告警/工单" 的完整闭环,
       并将结果缓存到内存数据仓(线程安全);
    2. REST API: 为前端(3D 场景 / 曲线图 / 告警列表)提供数据接口;
    3. 页面集成: 单页看板 = Three.js 3D 车间 + 实时传感器曲线 + 健康评分
       趋势 + 告警/工单列表 + 预测报告导出(Markdown 下载)。

启动:
    python app.py            # 访问 http://127.0.0.1:5000
================================================================================
"""

import os
import sys
import threading
import time
from collections import deque
from datetime import datetime
from urllib.parse import quote

from flask import Flask, jsonify, send_from_directory, Response

# ------------------------------------------------------------------------------
# 跨目录模块引入: data-ingestion / digital-twin / ai-prediction / alerts
# ------------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("data-ingestion", "digital-twin", "ai-prediction", "alerts"):
    sys.path.insert(0, os.path.join(PROJECT_ROOT, sub))

from data_simulator import DataSimulator, StreamWriter          # noqa: E402
from model_updater import TwinUpdater                           # noqa: E402
from predictive_engine import PredictiveEngine, DEFAULT_CSV     # noqa: E402
from fault_classifier import FaultClassifier                    # noqa: E402
from alert_engine import AlertEngine                            # noqa: E402

app = Flask(__name__)

# ------------------------------------------------------------------------------
# 内存数据仓: 看板轮询的唯一数据源(读写均持锁)
# ------------------------------------------------------------------------------
class DataStore:
    """线程安全的运行时状态仓。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.realtime = {}          # device_id -> deque(最近采样记录)
        self.health_hist = {}       # device_id -> deque(健康评分序列)
        self.predictions = {}       # device_id -> 最新预测结果
        self.faults = {}            # device_id -> 最新故障分类结果
        self.twin_summary = {}      # device_id -> 最新孪生同步摘要
        self.pipeline_ticks = 0     # 流水线累计轮次
        self.started_at = datetime.now().isoformat(timespec="seconds")

    def push(self, record, prediction, fault, twin_summary):
        """写入一轮流水线结果。"""
        with self.lock:
            did = record["device_id"]
            self.realtime.setdefault(did, deque(maxlen=120)).append(record)
            self.health_hist.setdefault(did, deque(maxlen=300)).append(
                prediction["health_score"])
            self.predictions[did] = prediction
            self.faults[did] = fault
            self.twin_summary[did] = twin_summary
            self.pipeline_ticks += 1

    def snapshot(self, device_id=None):
        """读取数据仓快照(可选按设备过滤)。"""
        with self.lock:
            return {
                "realtime": {k: list(v) for k, v in self.realtime.items()},
                "health_hist": {k: list(v) for k, v in self.health_hist.items()},
                "predictions": dict(self.predictions),
                "faults": dict(self.faults),
                "twin": dict(self.twin_summary),
                "ticks": self.pipeline_ticks,
                "started_at": self.started_at,
            }


store = DataStore()

# ------------------------------------------------------------------------------
# 后台流水线: 模拟 -> 孪生 -> 预测 -> 告警
# ------------------------------------------------------------------------------
class Pipeline:
    """演示流水线: 每 tick_seconds 推进一轮完整闭环。

    节奏说明: 生产系统中各环节由消息队列解耦; 演示环境合并为单线程
    顺序执行, 逻辑一致且便于观察数据流转。
    """

    # 演示加速: 每 tick_seconds(2.0) 真实秒推进 FAST_FORWARD 个模拟循环,
    # 使设备在数分钟内走完"健康 -> 故障"完整生命周期。
    # 注意: 模拟循环与 AI 喂入是两个概念——全部循环逐条喂给预测引擎(口径
    # 契约见 step()), 节流只体现在"展示层取最后一批"与孪生同步频率上。
    FAST_FORWARD = 6
    tick_seconds = 2.0

    def __init__(self):
        self.simulator = DataSimulator(seed=int(time.time()) % 10000)
        self.writer = StreamWriter()
        self.twin = TwinUpdater()
        self.engine = PredictiveEngine()
        self.classifier = FaultClassifier()
        self.alerts = AlertEngine()
        self._stop = threading.Event()

    def bootstrap(self):
        """初始化: 训练预测引擎(首次运行自动生成历史数据集)。"""
        if not os.path.exists(DEFAULT_CSV):
            print("[看板] 未发现历史数据集, 自动生成中 ...")
            from data_simulator import generate_historical_csv
            generate_historical_csv()
        print("[看板] 训练 AI 预测引擎 ...")
        summary = self.engine.train(DEFAULT_CSV)
        print("[看板] 训练完成: 样本 %d, sklearn 可用: %s"
              % (summary["total_samples"], summary["sklearn"]))

    def run(self):
        """流水线主循环。"""
        self.bootstrap()
        print("[看板] 流水线启动: 每 %.1f 秒推进 %d 个模拟循环"
              % (self.tick_seconds, self.FAST_FORWARD))
        while not self._stop.is_set():
            try:
                self.step()
            except Exception as exc:               # 单轮异常不终止流水线
                print("[看板] 流水线异常: %r" % exc)
            self._stop.wait(self.tick_seconds)

    def step(self):
        """推进一轮: 模拟 -> 存流 -> AI 逐条预测/告警 -> 孪生同步 -> 看板展示。

        喂入口径契约(v1.3 统一):
        FAST_FORWARD 个模拟循环产生的每一条记录都按"1 条记录 = 1 tick =
        10 分钟"逐条喂给预测引擎与告警引擎, 与 RULPredictor 的换算假设及
        evaluate.py 的评估喂入严格一致——看板在线 RUL、预测预警触发时刻与
        评估工件同口径(同一轨迹实测逐位一致)。v1.2 及之前只把最后一批喂给
        引擎, 同一设备状态的在线 RUL 与评估口径存在约 3 倍系统性分歧, 且
        预测预警在真值 RUL 远高于 72h 时即提前触发。
        数字孪生同步仍取最后一批: 孪生偏差按同步次数累积, 无小时量纲语义,
        且 update_all 每次调用都会落盘 twin_state.json, 保持演示节奏的同时
        控制写盘频率。
        """
        showcase = []                            # 最后一批(看板展示层取材)
        for k in range(self.FAST_FORWARD):
            records = self.simulator.step_all()
            self.writer.write(records)                       # 1) 数据落流
            for rec in records:                              # 2) AI 逐条喂入(口径契约)
                prediction = self.engine.assess(rec)         #    AI 预测
                fault = self.classifier.classify(            # 3) 故障归因
                    prediction["features"],
                    rec.get("device_type", "motor"),
                    health_score=prediction["health_score"])
                fired = self.alerts.evaluate(prediction)     # 4) 告警/工单
                if fired:
                    for a in fired:
                        print("[看板] 告警: %s" % a["title"])
                if k == self.FAST_FORWARD - 1:
                    showcase.append((rec, prediction, fault))
        twin_summaries = {s["device_id"]: s
                          for s in self.twin.update_all([r for r, _, _ in showcase])}
        for rec, prediction, fault in showcase:              # 5) 看板数据仓更新
            store.push(rec, prediction, fault,
                       twin_summaries.get(rec["device_id"], {}))

    def reset(self):
        """复位演示: 设备恢复全新状态, 清空孪生漂移/评分历史/活动告警。"""
        self.simulator.reset_all()
        self.twin.reset_all()                 # 孪生校准基线与偏差历史归零
        self.engine.health_history.clear()
        with store.lock:
            store.realtime.clear()            # 旧生命周期的实时曲线一并清空
            store.health_hist.clear()
            store.predictions.clear()
            store.faults.clear()
            store.twin_summary.clear()
            store.pipeline_ticks = 0
        # 告警复位走 AlertEngine.resolve_all(): 持锁改状态并落盘,
        # 避免与流水线线程的 evaluate() 竞态(互斥锁保护)
        self.alerts.resolve_all()
        print("[看板] 演示已复位为全新设备。")


pipeline = Pipeline()
pipeline_thread = threading.Thread(target=pipeline.run, daemon=True)


# ------------------------------------------------------------------------------
# REST API 路由
# ------------------------------------------------------------------------------
@app.route("/api/devices")
def api_devices():
    """全部设备最新状态(3D 场景与健康卡片的数据源)。"""
    snap = store.snapshot()
    devices = []
    for did, pred in snap["predictions"].items():
        latest = snap["realtime"][did][-1] if snap["realtime"].get(did) else {}
        devices.append({
            "device_id": did,
            "device_name": latest.get("device_name", did),
            "device_type": latest.get("device_type"),
            "health_score": pred["health_score"],
            "anomaly_score": pred["anomaly_score"],
            "stage": pred["stage"],
            "rul": pred["rul"],
            "vibration": latest.get("vibration"),
            "temperature": latest.get("temperature"),
            "current": latest.get("current"),
            "voltage": latest.get("voltage"),
            "rpm": latest.get("rpm"),
            "timestamp": pred.get("timestamp"),
        })
    return jsonify({"ticks": snap["ticks"], "devices": devices})


@app.route("/api/realtime/<device_id>")
def api_realtime(device_id):
    """单设备最近 N 条实时采样(实时曲线数据源)。"""
    snap = store.snapshot()
    series = snap["realtime"].get(device_id, [])
    return jsonify({"device_id": device_id, "count": len(series), "series": series})


@app.route("/api/health/<device_id>")
def api_health(device_id):
    """单设备健康评分历史序列(健康趋势曲线数据源)。"""
    snap = store.snapshot()
    return jsonify({"device_id": device_id,
                    "history": snap["health_hist"].get(device_id, [])})


@app.route("/api/prediction/<device_id>")
def api_prediction(device_id):
    """单设备最新预测报告(3D 详情面板数据源)。"""
    snap = store.snapshot()
    pred = snap["predictions"].get(device_id)
    fault = snap["faults"].get(device_id)
    twin = snap["twin"].get(device_id)
    if pred is None:
        return jsonify({"error": "设备尚未产生预测数据"}), 404
    return jsonify({"prediction": pred, "fault": fault, "twin": twin})


@app.route("/api/twin")
def api_twin():
    """数字孪生同步摘要(偏差/同步度)。"""
    return jsonify({"twins": pipeline.twin.snapshots()})


@app.route("/api/alerts")
def api_alerts():
    """活动告警列表(按时间倒序)。"""
    return jsonify({"alerts": pipeline.alerts.active_alerts()})


@app.route("/api/workorders")
def api_workorders():
    """未关闭工单列表。"""
    return jsonify({"workorders": pipeline.alerts.open_workorders()})


@app.route("/api/workorders/<workorder_id>/close", methods=["POST"])
def api_close_workorder(workorder_id):
    """关闭工单(维修完成确认)。"""
    ok = pipeline.alerts.close_workorder(workorder_id)
    return jsonify({"closed": ok}), (200 if ok else 404)


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """复位演示(设备恢复全新)。"""
    pipeline.reset()
    return jsonify({"reset": True})


@app.route("/api/export/report")
def api_export_report():
    """导出全部设备的预测报告(Markdown 文本下载)。"""
    snap = store.snapshot()
    md = ["# 设备智能运维预测报告", "",
          "- 导出时间: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
          "- 数据流水线轮次: %d (看板启动于 %s)" % (snap["ticks"], snap["started_at"]),
          "- 预测后端: %s" % (pipeline.engine.backend), ""]
    for did, pred in snap["predictions"].items():
        rul = pred["rul"]
        fault = snap["faults"].get(did, {})
        twin = snap["twin"].get(did, {})
        md.append("## %s" % did)
        md.append("")
        md.append("| 指标 | 数值 |")
        md.append("| --- | --- |")
        md.append("| 健康评分 | %.1f / 100 |" % pred["health_score"])
        md.append("| 异常评分 | %.1f / 100 |" % pred["anomaly_score"])
        md.append("| 生命周期阶段 | %s |" % pred["stage"])
        if rul.get("rul_hours") is not None:
            md.append("| 剩余寿命 RUL | %.1f 小时 (80%%CI: %.1f ~ %.1f) |"
                      % (rul["rul_hours"], rul["rul_ci_low"], rul["rul_ci_high"]))
            md.append("| 退化速率 | %.2f 分/天, 趋势 %s |"
                      % (rul.get("decline_per_day", 0), rul.get("trend")))
        if twin:
            md.append("| 孪生同步度 | %.1f |" % twin.get("sync_score", "-"))
        md.append("")
        if fault and not fault.get("ruled_out"):
            top = fault["matched"][0]
            md.append("**疑似故障**: %s(%s, 置信度 %.0f%%)"
                      % (top["name"], top["code"], top["confidence"] * 100))
            md.append("")
            md.append("处置建议:")
            for i, act in enumerate(fault["matched"][0]["actions"], 1):
                md.append("%d. %s" % (i, act))
            if top.get("spare_parts"):
                md.append("")
                md.append("备件清单: %s" % ", ".join(top["spare_parts"]))
        else:
            md.append("未检出显著故障特征, 维持常规监测策略。")
        md.append("")
    alerts = pipeline.alerts.active_alerts()
    if alerts:
        md.append("## 当前活动告警")
        md.append("")
        md.append("| 时间 | 设备 | 级别 | 内容 |")
        md.append("| --- | --- | --- | --- |")
        for a in alerts:
            md.append("| %s | %s | %s | %s |"
                      % (a["ts"], a["device_id"], a["severity_name"], a["title"]))
        md.append("")
    body = "\n".join(md)
    # HTTP 头仅支持 latin-1: 中文名须经 URL 编码(filename*=UTF-8''RFC5987 语法)
    filename = "预测报告_%s.md" % datetime.now().strftime("%Y%m%d_%H%M%S")
    return Response(body, mimetype="text/markdown; charset=utf-8",
                    headers={"Content-Disposition":
                             "attachment; filename*=UTF-8''%s" % quote(filename)})


@app.route("/3d/app.js")
def serve_3d_app():
    """把 visualization/3d_scene/app.js 挂载到看板(同源加载, 避免跨域)。"""
    return send_from_directory(
        os.path.join(PROJECT_ROOT, "visualization", "3d_scene"), "app.js")


@app.route("/")
def index():
    """看板单页(3D 场景 + 曲线 + 告警)。"""
    return INDEX_HTML


# ------------------------------------------------------------------------------
# 看板页面模板(内嵌单页应用: Chart.js 曲线 + Three.js 3D 场景)
# ------------------------------------------------------------------------------
INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>数字孪生智能运维看板</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: "Microsoft YaHei", sans-serif; background: #0d1420; color: #e8f1fa; }
  header { display: flex; justify-content: space-between; align-items: center;
           padding: 10px 20px; background: #101a2b; border-bottom: 1px solid #22344e; }
  header h1 { font-size: 18px; }
  header .meta { font-size: 12px; color: #7fa8c9; }
  main { display: grid; grid-template-columns: 1.4fr 1fr; gap: 12px; padding: 12px; }
  .card { background: #111c2e; border: 1px solid #22344e; border-radius: 10px; padding: 12px; }
  .card h2 { font-size: 14px; margin-bottom: 8px; color: #9fc3e2; }
  #twin-3d-container { position: relative; height: 520px; border-radius: 8px; overflow: hidden; }
  .btn { background: #1d3350; border: 1px solid #3d5a80; color: #cfe3f7; border-radius: 6px;
         padding: 5px 14px; cursor: pointer; font-size: 12px; margin-left: 8px; }
  .btn:hover { background: #27476e; }
  canvas { max-height: 210px; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th, td { padding: 5px 6px; border-bottom: 1px solid #22344e; text-align: left; }
  tr:hover { background: #16243c; }
  .sev-warning { color: #f1c40f; } .sev-predictive { color: #3498db; }
  .sev-critical { color: #e67e22; } .sev-emergency { color: #e74c3c; font-weight: bold; }
  select { background: #1d3350; color: #cfe3f7; border: 1px solid #3d5a80;
           border-radius: 6px; padding: 4px 8px; }
  .health-cell { font-weight: bold; }
  .row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  #alerts-card { max-height: 300px; overflow-y: auto; }
</style>
</head>
<body>
<header>
  <h1>数字孪生 · 设备智能运维看板</h1>
  <div>
    <span class="meta" id="tick-info">流水线启动中 ...</span>
    <button class="btn" onclick="window.open('/api/export/report')">导出预测报告</button>
    <button class="btn" onclick="fetch('/api/reset',{method:'POST'}).then(refreshAll)">复位演示</button>
  </div>
</header>
<main>
  <!-- 左列: 3D 车间 + 实时曲线 -->
  <div>
    <div class="card" style="margin-bottom:12px">
      <h2>3D 数字孪生车间 <span style="color:#5c7ea0;font-weight:normal">(拖拽旋转 / 滚轮缩放 / 点击设备查看详情; 颜色=健康评分)</span></h2>
      <div id="twin-3d-container"></div>
    </div>
    <div class="card">
      <h2>实时传感器曲线
        <select id="device-select" onchange="onDeviceChange()"></select>
      </h2>
      <canvas id="realtime-chart"></canvas>
    </div>
  </div>
  <!-- 右列: 健康趋势 + 告警 -->
  <div>
    <div class="card" style="margin-bottom:12px">
      <h2>健康评分趋势 (0~100)</h2>
      <canvas id="health-chart"></canvas>
    </div>
    <div class="card" style="margin-bottom:12px">
      <h2>设备健康总览</h2>
      <table id="devices-table">
        <thead><tr><th>设备</th><th>健康评分</th><th>异常分</th><th>RUL(小时)</th><th>阶段</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
    <div class="card" id="alerts-card">
      <h2>活动告警与工单</h2>
      <table id="alerts-table">
        <thead><tr><th>时间</th><th>设备</th><th>级别</th><th>内容</th><th>工单</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>
</main>

<script src="/3d/app.js"></script>
<script>
/* ------------------------------ 看板前端逻辑 ------------------------------ */
let realtimeChart = null, healthChart = null;
let selectedDevice = 'MOTOR-001';
const CHANNELS = ['vibration', 'temperature', 'current', 'voltage'];
const CHANNEL_NAMES = { vibration: '振动(mm/s)', temperature: '温度(℃)',
                        current: '电流(A)', voltage: '电压(V)' };

function healthColor(score) {
  return score >= 80 ? '#2ecc71' : score >= 60 ? '#f1c40f'
       : score >= 40 ? '#e67e22' : '#e74c3c';
}

function onDeviceChange() {
  selectedDevice = document.getElementById('device-select').value;
  refreshAll();
}

/* 刷新设备选择下拉框与总览表 */
async function refreshDevices() {
  const data = await (await fetch('/api/devices')).json();
  const sel = document.getElementById('device-select');
  data.devices.forEach(d => {
    if (!Array.from(sel.options).some(o => o.value === d.device_id)) {
      sel.add(new Option(d.device_name + '(' + d.device_id + ')', d.device_id));
    }
  });
  document.getElementById('tick-info').textContent =
    '流水线轮次: ' + data.ticks + ' · 更新: ' + new Date().toLocaleTimeString();
  const tbody = document.querySelector('#devices-table tbody');
  tbody.innerHTML = data.devices.map(d => {
    const rul = d.rul && d.rul.rul_hours !== null && d.rul.rul_hours !== undefined
      ? d.rul.rul_hours.toFixed(1) : '-';
    return '<tr><td>' + d.device_name + '<br><small style="color:#5c7ea0">' + d.device_id + '</small></td>'
      + '<td class="health-cell" style="color:' + healthColor(d.health_score) + '">' + d.health_score.toFixed(1) + '</td>'
      + '<td>' + d.anomaly_score.toFixed(0) + '</td><td>' + rul + '</td><td>' + d.stage + '</td></tr>';
  }).join('');
}

/* 刷新实时传感器曲线(多通道双轴) */
async function refreshRealtime() {
  const data = await (await fetch('/api/realtime/' + selectedDevice)).json();
  const labels = data.series.map(r => r.timestamp.slice(5, 16));
  if (!realtimeChart) {
    realtimeChart = new Chart(document.getElementById('realtime-chart'), {
      type: 'line',
      data: { labels, datasets: CHANNELS.map(ch => ({
        label: CHANNEL_NAMES[ch], data: data.series.map(r => r[ch]),
        borderWidth: 1.5, pointRadius: 0, tension: 0.3, yAxisID: ch === 'voltage' ? 'y1' : 'y' })) },
      options: { animation: false, responsive: true,
        plugins: { legend: { labels: { color: '#9fc3e2', boxWidth: 20, font: { size: 10 } } } },
        scales: {
          x: { ticks: { color: '#5c7ea0', maxTicksLimit: 8, font: { size: 10 } }, grid: { color: '#1a2a44' } },
          y: { position: 'left', ticks: { color: '#5c7ea0', font: { size: 10 } }, grid: { color: '#1a2a44' } },
          y1: { position: 'right', ticks: { color: '#5c7ea0', font: { size: 10 } }, grid: { drawOnChartArea: false } }
        } }
    });
  } else {
    realtimeChart.data.labels = labels;
    CHANNELS.forEach((ch, i) => { realtimeChart.data.datasets[i].data = data.series.map(r => r[ch]); });
    realtimeChart.update('none');
  }
}

/* 刷新健康评分趋势曲线 */
async function refreshHealth() {
  const data = await (await fetch('/api/health/' + selectedDevice)).json();
  const labels = data.history.map((_, i) => i);
  const coloring = ctx => healthColor(data.history[ctx.dataIndex] ?? 100);
  if (!healthChart) {
    healthChart = new Chart(document.getElementById('health-chart'), {
      type: 'line',
      data: { labels, datasets: [{ label: '健康评分', data: data.history,
        borderWidth: 2, pointRadius: 0, tension: 0.3, fill: true,
        backgroundColor: 'rgba(46,204,113,0.08)' }] },
      options: { animation: false, responsive: true,
        plugins: { legend: { display: false } },
        scales: { x: { ticks: { color: '#5c7ea0', maxTicksLimit: 6, font: { size: 10 } }, grid: { color: '#1a2a44' } },
                  y: { min: 0, max: 100, ticks: { color: '#5c7ea0', font: { size: 10 } }, grid: { color: '#1a2a44' } } } }
    });
    healthChart.data.datasets[0].segment = { borderColor: ctx => coloring(ctx) };
  } else {
    healthChart.data.labels = labels;
    healthChart.data.datasets[0].data = data.history;
    healthChart.update('none');
  }
}

/* 刷新告警与工单表 */
async function refreshAlerts() {
  const [alertData, woData] = await Promise.all([
    fetch('/api/alerts').then(r => r.json()),
    fetch('/api/workorders').then(r => r.json())
  ]);
  const woByAlert = {};
  woData.workorders.forEach(w => { if (w.alert_id) woByAlert[w.alert_id] = w; });
  const tbody = document.querySelector('#alerts-table tbody');
  const rows = alertData.alerts.map(a => {
    const wo = woByAlert[a.id];
    const woCell = wo ? '<button class="btn" style="padding:2px 8px" onclick="closeWO(\\'' + wo.id + '\\')">' + wo.id + ' 完成维修</button>' : '-';
    return '<tr><td>' + a.ts.slice(5, 16) + '</td><td>' + a.device_id + '</td>'
      + '<td class="sev-' + a.severity + '">' + a.severity_name + '</td>'
      + '<td title="' + (a.detail || '') + '">' + a.title + '</td><td>' + woCell + '</td></tr>';
  });
  tbody.innerHTML = rows.join('') || '<tr><td colspan="5" style="color:#5c7ea0">暂无活动告警 —— 设备均运行在安全区间</td></tr>';
}

async function closeWO(id) {
  await fetch('/api/workorders/' + id + '/close', { method: 'POST' });
  refreshAlerts();
}

function refreshAll() {
  refreshDevices(); refreshRealtime(); refreshHealth(); refreshAlerts();
}
refreshAll();
setInterval(refreshAll, 3000);
</script>
</body>
</html>
"""


# ------------------------------------------------------------------------------
# 启动入口
# ------------------------------------------------------------------------------
def main():
    """启动看板: 先拉起后台流水线线程, 再启动 Flask 服务。"""
    pipeline_thread.start()
    print("[看板] 访问地址: http://127.0.0.1:5000  (Ctrl+C 退出)")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()
