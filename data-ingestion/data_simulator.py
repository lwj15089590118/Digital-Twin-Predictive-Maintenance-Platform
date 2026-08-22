# -*- coding: utf-8 -*-
"""
================================================================================
 数据模拟器 (Data Simulator)
================================================================================
模块职责:
    1. 模拟车间三台关键设备(主电机 / 离心风机 / 齿轮传动箱)的实时传感器数据,
       通道包括: 振动(mm/s)、温度(℃)、电流(A)、电压(V)、转速(rpm);
    2. 内置"健康度衰退模型": 设备健康度 h(t) 从 1.0(健康) 随运行循环单调衰退到 0.0
       (故障), 传感器读数随健康度恶化呈现不同的物理响应(振动上升、温升加大、
       电流增加、转速下降等), 完整模拟"健康 -> 轻度衰退 -> 明显退化 -> 严重退化
       -> 临故障"的全生命周期;
    3. 支持故障注入(轴承磨损 / 转子不平衡 / 润滑不足 / 齿轮点蚀等), 用于生成
       带标签的历史训练数据;
    4. 输出方式:
       - 流式模式: 每个采样周期向 data/stream/ 目录追加 JSONL 数据流,
         并刷新 latest.json 快照供数字孪生与看板消费;
       - 历史模式: 一次性生成历史训练数据集 CSV(供 AI 预测引擎训练)。

运行示例:
    python data_simulator.py --mode csv --rows-per-device 220   # 生成历史数据集
    python data_simulator.py --mode stream --interval 1.0       # 启动实时数据流
================================================================================
"""

import argparse
import csv
import json
import math
import os
import random
import time
from datetime import datetime, timedelta

# ------------------------------------------------------------------------------
# 全局常量定义
# ------------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STREAM_DIR = os.path.join(PROJECT_ROOT, "data", "stream")       # 实时数据流目录
DEFAULT_CSV_PATH = os.path.join(PROJECT_ROOT, "data-ingestion", "historical_data.csv")

SEED = 2026                    # 随机种子: 保证历史数据可复现
SIM_TICK_MINUTES = 10          # 每个模拟循环代表的真实时间(分钟), 用于换算 RUL

# ------------------------------------------------------------------------------
# 设备台账(设备注册表)
# ------------------------------------------------------------------------------
DEVICES = {
    "MOTOR-001": {
        "name": "主电机", "type": "motor", "rated_power_kw": 45.0,
        "rated_rpm": 1480.0,
        # 健康状态下的基线传感器读数
        "baseline": {"vibration": 1.6, "temperature": 55.0, "current": 82.0, "voltage": 380.0},
        # 全生命周期总循环数(衰退到 0 所需循环)
        "life_cycles": 1320,
        # 该设备可能的故障模式(用于故障注入与标签生成)
        "fault_modes": ["bearing_wear", "winding_insulation"],
    },
    "FAN-001": {
        "name": "离心风机", "type": "fan", "rated_power_kw": 30.0,
        "rated_rpm": 980.0,
        "baseline": {"vibration": 2.4, "temperature": 48.0, "current": 56.0, "voltage": 380.0},
        "life_cycles": 1560,
        "fault_modes": ["rotor_imbalance", "bearing_wear"],
    },
    "GEARBOX-001": {
        "name": "齿轮传动箱", "type": "gearbox", "rated_power_kw": 40.0,
        "rated_rpm": 1250.0,
        "baseline": {"vibration": 2.0, "temperature": 62.0, "current": 74.0, "voltage": 380.0},
        "life_cycles": 1440,
        "fault_modes": ["gear_pitting", "lubrication_failure"],
    },
}

# 故障模式中英对照表(写 CSV 时输出可读的故障名称)
FAULT_MODE_NAMES = {
    "bearing_wear": "轴承磨损",
    "winding_insulation": "绕组绝缘老化",
    "rotor_imbalance": "转子不平衡",
    "gear_pitting": "齿轮点蚀",
    "lubrication_failure": "润滑不足",
}


# ------------------------------------------------------------------------------
# 健康度 -> 生命周期阶段 映射
# ------------------------------------------------------------------------------
def health_stage(health: float) -> tuple:
    """根据健康度(0~1)返回 (阶段中文名, 机器学习标签)。

    标签约定:
        normal  - 健康度 > 0.8, 正常运行
        warning - 0.5 < 健康度 <= 0.8, 性能退化前兆(预测性维护黄金窗口)
        fault   - 健康度 <= 0.5, 严重退化/临故障, 需立即干预
    """
    if health > 0.8:
        return "正常运行", "normal"
    elif health > 0.6:
        return "轻度衰退", "warning"
    elif health > 0.4:
        return "明显退化", "fault"
    elif health > 0.2:
        return "严重退化", "fault"
    return "临故障", "fault"


# ------------------------------------------------------------------------------
# 单台设备模拟器
# ------------------------------------------------------------------------------
class DeviceSimulator:
    """单台设备的物理退化仿真器。

    核心思想:
        - 健康度 h(t) 按幂函数曲线单调衰退: h = 1 - (t/T)^1.6, T 为寿命循环数;
          幂指数 1.6 模拟"先缓后急"的典型浴盆曲线后段(磨损累积加速);
        - 各传感器通道按各自的物理规律响应健康度:
            振动   vib  = base * (1 + k_v * (1-h)^1.7)   轴承磨损 -> 振动显著上升
            温度   temp = base + k_t * (1-h)^1.3          摩擦生热 -> 温升
            电流   cur  = base * (1 + k_c * (1-h))        效率下降 -> 电流上升
            电压   vol  = 380 - k_d * (1-h)               重载下电网压降
            转速   rpm  = rated * (1 - k_r * (1-h))       负载/故障导致掉速
        - 故障注入: 当健康度越过注入阈值后, 叠加对应故障的特征增量
          (例如轴承磨损带来周期性振动冲击尖峰)。
    """

    def __init__(self, device_id: str, config: dict, rng: random.Random):
        self.device_id = device_id
        self.config = config
        self.rng = rng
        self.cycle = 0                       # 当前运行循环(从 0 开始)
        self.life_cycles = config["life_cycles"]
        self.baseline = config["baseline"]
        self.injections = []                 # 已注入的故障列表 [(阈值, 模式, 强度)]
        self.start_time = datetime(2026, 1, 1, 8, 0, 0)   # 模拟时钟起点
        # 随机工况扰动: 每台设备拥有缓慢变化的负载因子(0.85~1.15)
        self._load_factor = 1.0

    # ---------------------------- 健康度模型 ----------------------------
    def health(self) -> float:
        """计算当前循环的健康度(真值, 仅模拟器可知, 真实系统中由 AI 引擎估计)。"""
        if self.cycle >= self.life_cycles:
            return 0.0
        ratio = self.cycle / self.life_cycles
        h = 1.0 - ratio ** 1.6
        # 叠加小幅测量级扰动, 避免曲线过于理想化
        h += self.rng.gauss(0, 0.008)
        return max(0.0, min(1.0, h))

    # ---------------------------- 工况扰动 ----------------------------
    def _update_load(self):
        """负载因子做缓慢随机游走, 模拟产线负荷波动。"""
        self._load_factor += self.rng.gauss(0, 0.02)
        self._load_factor = max(0.85, min(1.15, self._load_factor))

    def inject_fault(self, threshold: float, mode: str, strength: float = 1.0):
        """注册一个故障注入: 当健康度跌破 threshold 时激活对应故障模式。

        Args:
            threshold: 激活阈值(健康度), 例如 0.55 表示轻度衰退后期开始出现故障特征
            mode:      故障模式键名(见 FAULT_MODE_NAMES)
            strength:  故障强度系数, 1.0 为标准强度
        """
        self.injections.append({"threshold": threshold, "mode": mode, "strength": strength})

    def _active_faults(self, h: float) -> list:
        """返回当前健康度下已激活的故障模式列表。"""
        return [f for f in self.injections if h <= f["threshold"]]

    # ---------------------------- 传感器模型 ----------------------------
    def _vibration(self, h: float, faults: list) -> float:
        """振动速度有效值(mm/s): 退化越严重振动越大, 轴承类故障叠加冲击尖峰。"""
        base = self.baseline["vibration"]
        value = base * (1.0 + 3.2 * (1.0 - h) ** 1.7) * self._load_factor
        for f in faults:
            if f["mode"] in ("bearing_wear", "gear_pitting"):
                # 轴承磨损/齿轮点蚀: 周期性冲击, 振动叠加 30%~70% 尖峰
                spike = 0.3 + 0.4 * (1.0 - h)
                value *= 1.0 + spike * f["strength"] * (1.0 + 0.5 * math.sin(self.cycle / 7.0))
            elif f["mode"] == "rotor_imbalance":
                # 转子不平衡: 振幅与转速平方近似成正比, 持续性抬升
                value *= 1.0 + 0.55 * (1.0 - h) * f["strength"]
        return max(0.1, value * (1.0 + self.rng.gauss(0, 0.05)))

    def _temperature(self, h: float, faults: list) -> float:
        """绕组/油温(℃): 摩擦损耗增大 -> 温升; 润滑不足与绝缘老化显著抬高温度。"""
        base = self.baseline["temperature"]
        value = base + 30.0 * (1.0 - h) ** 1.3
        value += 4.0 * (self._load_factor - 1.0)          # 负载越高温升越大
        # 日周期环境温度波动(车间昼夜温差约 ±3℃)
        value += 3.0 * math.sin(2 * math.pi * self.cycle / 144.0)
        for f in faults:
            if f["mode"] in ("lubrication_failure", "winding_insulation"):
                value += 12.0 * (1.0 - h) * f["strength"]
        return value + self.rng.gauss(0, 0.6)

    def _current(self, h: float, faults: list) -> float:
        """工作电流(A): 效率退化 -> 电流上升, 过载工况再叠加。"""
        base = self.baseline["current"]
        value = base * (1.0 + 0.22 * (1.0 - h)) * self._load_factor
        for f in faults:
            if f["mode"] == "winding_insulation":
                value *= 1.0 + 0.12 * (1.0 - h) * f["strength"]
        return max(1.0, value * (1.0 + self.rng.gauss(0, 0.02)))

    def _voltage(self, h: float) -> float:
        """母线电压(V): 围绕 380V 小幅波动, 重载退化阶段略有压降。"""
        value = 380.0 - 6.0 * (1.0 - h) + self.rng.gauss(0, 3.5)
        return max(340.0, min(420.0, value))

    def _rpm(self, h: float, faults: list) -> float:
        """转速(rpm): 退化导致转差率增大, 不平衡故障引起转速小幅波动加大。"""
        rated = self.config["rated_rpm"]
        value = rated * (1.0 - 0.06 * (1.0 - h))
        jitter = 6.0
        for f in faults:
            if f["mode"] == "rotor_imbalance":
                jitter = 14.0
        return max(0.0, value + self.rng.gauss(0, jitter))

    # ---------------------------- 单步仿真 ----------------------------
    def step(self) -> dict:
        """推进一个仿真循环, 返回一条完整的传感器采样记录(dict)。"""
        self._update_load()
        h = self.health()
        faults = self._active_faults(h)
        record = {
            "device_id": self.device_id,
            "device_name": self.config["name"],
            "device_type": self.config["type"],
            "cycle": self.cycle,
            # 模拟时钟: 起点 + 循环数 * 每循环分钟数
            "timestamp": (self.start_time + timedelta(minutes=self.cycle * SIM_TICK_MINUTES)).isoformat(),
            "vibration": round(self._vibration(h, faults), 3),
            "temperature": round(self._temperature(h, faults), 2),
            "current": round(self._current(h, faults), 2),
            "voltage": round(self._voltage(h), 2),
            "rpm": round(self._rpm(h, faults), 1),
            # 真实健康度与标签(仅用于训练/评估, 预测引擎不使用 health 字段)
            "health": round(h, 4),
            "fault_modes": [f["mode"] for f in faults],
        }
        record["stage"], record["label"] = health_stage(h)
        self.cycle += 1
        return record

    def reset(self):
        """复位到全新设备状态(健康度 1.0)。"""
        self.cycle = 0
        self._load_factor = 1.0


# ------------------------------------------------------------------------------
# 多设备协同仿真器
# ------------------------------------------------------------------------------
class DataSimulator:
    """管理全部设备的协同仿真, 并提供数据流输出能力。"""

    def __init__(self, seed: int = SEED):
        self.rng = random.Random(seed)
        self.devices = {}
        for device_id, cfg in DEVICES.items():
            sim = DeviceSimulator(device_id, cfg, random.Random(seed + hash(device_id) % 1000))
            # 为每台设备注册典型的故障注入计划(阈值即健康度越过点)
            plans = {
                "bearing_wear": 0.60,       # 轴承磨损: 轻度衰退后期出现
                "winding_insulation": 0.45, # 绕组绝缘老化: 明显退化期出现
                "rotor_imbalance": 0.55,    # 转子不平衡(叶轮积尘结垢典型后果)
                "gear_pitting": 0.50,       # 齿轮点蚀
                "lubrication_failure": 0.40,# 润滑不足
            }
            for mode in cfg["fault_modes"]:
                sim.inject_fault(plans.get(mode, 0.5), mode, strength=1.0)
            self.devices[device_id] = sim

    def step_all(self) -> list:
        """所有设备各推进一个循环, 返回本轮全部采样记录。"""
        return [sim.step() for sim in self.devices.values()]

    def reset_all(self):
        """全部设备复位(用于演示新一轮生命周期)。"""
        for sim in self.devices.values():
            sim.reset()


# ------------------------------------------------------------------------------
# 数据流写入器: JSONL 追加 + 最新快照
# ------------------------------------------------------------------------------
class StreamWriter:
    """将采样记录写入本地数据流文件。

    文件布局:
        data/stream/device_<id>.jsonl   每台设备一条 JSON Lines 时间序列(滚动截断)
        data/stream/latest.json         全部设备最新一条记录的快照(便于看板轮询)
    """

    MAX_LINES = 500   # 每台设备保留的最大历史行数(防止文件无限增长)

    def __init__(self, stream_dir: str = STREAM_DIR):
        self.stream_dir = stream_dir
        os.makedirs(stream_dir, exist_ok=True)
        self.latest_path = os.path.join(stream_dir, "latest.json")

    def write(self, records: list):
        """写入一批采样记录: 追加 JSONL 并刷新快照。"""
        latest = {}
        if os.path.exists(self.latest_path):
            try:
                with open(self.latest_path, "r", encoding="utf-8") as f:
                    latest = json.load(f)
            except (json.JSONDecodeError, OSError):
                latest = {}
        for rec in records:
            path = os.path.join(self.stream_dir, "device_%s.jsonl" % rec["device_id"])
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            latest[rec["device_id"]] = rec          # 快照保留每台设备最新值
        with open(self.latest_path, "w", encoding="utf-8") as f:
            json.dump(latest, f, ensure_ascii=False, indent=2)
        self._rotate()

    def _rotate(self):
        """滚动截断: 超过最大行数时保留尾部, 防止演示环境磁盘膨胀。"""
        for filename in os.listdir(self.stream_dir):
            if not filename.endswith(".jsonl"):
                continue
            path = os.path.join(self.stream_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) > self.MAX_LINES:
                with open(path, "w", encoding="utf-8") as f:
                    f.writelines(lines[-self.MAX_LINES:])


# ------------------------------------------------------------------------------
# 历史数据集生成(用于训练预测模型)
# ------------------------------------------------------------------------------
def generate_historical_csv(csv_path: str = DEFAULT_CSV_PATH, rows_per_device: int = 220) -> str:
    """生成历史训练数据集 CSV。

    做法: 每台设备完整仿真一遍生命周期(life_cycles 个循环),
    每隔 sample_step 个循环抽取一条记录, 保证覆盖
    "正常运行 -> 轻度衰退 -> 明显退化 -> 严重退化 -> 临故障" 全部阶段。

    CSV 列:
        timestamp, device_id, device_name, cycle, vibration, temperature,
        current, voltage, rpm, health, stage, label, fault_type
    其中 label(normal/warning/fault) 与 fault_type(中文故障名) 为监督标签。
    """
    sim = DataSimulator(seed=SEED)
    header = ["timestamp", "device_id", "device_name", "cycle", "vibration",
              "temperature", "current", "voltage", "rpm", "health", "stage",
              "label", "fault_type"]
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    total = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for device_id, dev in sim.devices.items():
            life = dev.life_cycles
            # 采样步长: 保证抽满 rows_per_device 条左右
            sample_step = max(1, life // rows_per_device)
            for _ in range(life + sample_step):
                rec = dev.step()
                if rec["cycle"] % sample_step == 0:
                    # 故障类型列: 列出该时刻全部已激活的故障(以 "+" 连接)
                    fault_type = "+".join(FAULT_MODE_NAMES[m]
                                          for m in rec["fault_modes"])
                    writer.writerow([rec["timestamp"], device_id, rec["device_name"],
                                     rec["cycle"], rec["vibration"], rec["temperature"],
                                     rec["current"], rec["voltage"], rec["rpm"],
                                     rec["health"], rec["stage"], rec["label"], fault_type])
                    total += 1
    return csv_path


# ------------------------------------------------------------------------------
# 流式运行主循环
# ------------------------------------------------------------------------------
def run_stream(interval: float = 1.0, cycles: int = 0):
    """以流式模式持续运行: 每 interval 秒产生一轮数据并写盘。

    Args:
        interval: 采样周期(秒, 演示加速; 真实系统为 10 分钟)
        cycles:   总轮数, 0 表示无限运行
    """
    sim = DataSimulator()
    writer = StreamWriter()
    print("[数据模拟器] 启动实时数据流, 输出目录: %s" % writer.stream_dir)
    i = 0
    try:
        while cycles == 0 or i < cycles:
            records = sim.step_all()
            writer.write(records)
            for r in records:
                print("[tick %4d] %s 振动=%.2f 温度=%.1f 电流=%.1f 健康=%.2f (%s)"
                      % (i, r["device_id"], r["vibration"], r["temperature"],
                         r["current"], r["health"], r["stage"]))
            # 所有设备生命周期结束后自动复位, 演示可无限循环
            if all(d.cycle >= d.life_cycles for d in sim.devices.values()):
                print("[数据模拟器] 本轮生命周期结束, 10 秒后自动复位重启 ...")
                time.sleep(10)
                sim.reset_all()
            i += 1
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[数据模拟器] 已手动停止。")


# ------------------------------------------------------------------------------
# 命令行入口
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="数字孪生平台 - 设备数据模拟器")
    parser.add_argument("--mode", choices=["stream", "csv"], default="stream",
                        help="运行模式: stream=实时数据流, csv=生成历史训练数据")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="流式模式的采样周期(秒)")
    parser.add_argument("--cycles", type=int, default=0,
                        help="流式模式总轮数(0=无限)")
    parser.add_argument("--rows-per-device", type=int, default=220,
                        help="CSV 模式下每台设备抽取的样本行数")
    parser.add_argument("--output", default=DEFAULT_CSV_PATH,
                        help="CSV 输出路径")
    args = parser.parse_args()

    if args.mode == "csv":
        path = generate_historical_csv(args.output, args.rows_per_device)
        # 统计生成的样本数, 便于核对
        with open(path, "r", encoding="utf-8") as f:
            n = sum(1 for _ in f) - 1
        print("[数据模拟器] 历史数据集已生成: %s (共 %d 条样本)" % (path, n))
    else:
        run_stream(args.interval, args.cycles)


if __name__ == "__main__":
    main()
