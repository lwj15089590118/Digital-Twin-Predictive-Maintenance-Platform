# -*- coding: utf-8 -*-
"""
================================================================================
 故障分类器 (Fault Classifier)
================================================================================
模块职责:
    根据设备的异常模式(各传感器通道相对基线的偏离方向与幅度), 在内置
    "故障知识库"中匹配最可能的故障类型, 并给出对应的处理方案与建议措施。

匹配原理(加权证据评分):
    每种故障在知识库中登记了一组"特征签名"(signature), 例如:
        轴承磨损   -> 振动显著上升 + 温度中度上升 + 电流轻度上升
        转子不平衡 -> 振动上升(转速相关) + 其余通道基本正常
        润滑不足   -> 温度显著上升 + 振动中度上升
    分类时将实时数据的 z 分数与签名逐通道比对, 命中(方向一致且幅度
    达到证据阈值)则累加该故障的置信分, 最终按总分排序输出 Top-K。

    该规则库可与数据驱动模型(随机森林等)互补: 规则给出可解释的物理
    归因, 模型给出统计置信, 两者结合形成"AI + 专家经验"的混合诊断。

运行示例:
    python fault_classifier.py --selftest    # 内置用例自检
================================================================================
"""

import argparse
import json


def clip01(value: float) -> float:
    """将数值裁剪到 [0, 1] 区间(证据强度的通用工具)。"""
    return max(0.0, min(1.0, value))

# ------------------------------------------------------------------------------
# 故障知识库 (Fault Library)
# ------------------------------------------------------------------------------
# signature 字段: {通道: (期望z区间下限, 期望z区间上限, 证据权重)}
#   - z 区间描述该故障激活时此通道应有的标准化偏离范围;
#   - 实测 z 落入区间即按权重得分; 越界(方向不符)适度扣分;
#   - 权重之和反映该故障的整体可检性。
FAULT_LIBRARY = [
    {
        "code": "F001",
        "name": "轴承磨损",
        "component": "滚动轴承",
        "applicable_devices": ["motor", "fan", "gearbox"],
        "signature": {
            "vibration": (2.5, 99.0, 3.0),     # 振动显著上升(冲击特征)
            "temperature": (1.0, 6.0, 1.5),    # 摩擦温升(中度)
            "current": (0.0, 3.0, 0.5),        # 电流轻度上升
        },
        "severity": "high",
        "risk_of_failure_hours": 168,
        "symptoms": "轴承部位温升加快, 振动频谱出现轴承特征频率冲击, 伴随周期性异响",
        "actions": [
            "测量轴承座振动频谱, 确认轴承特征频率(内圈/外圈/滚动体)",
            "检查润滑脂状态, 补充或更换同型号润滑脂",
            "测量轴承游隙与温升, 评估磨损等级",
            "磨损达 III 级以上时安排更换轴承, 建议同时更换油封",
        ],
        "spare_parts": ["轴承 6216-2Z x2", "润滑脂 Shell Gadus S2 V220 x1kg"],
    },
    {
        "code": "F002",
        "name": "转子不平衡",
        "component": "转子/叶轮",
        "applicable_devices": ["fan", "motor"],
        "signature": {
            "vibration": (1.8, 99.0, 2.5),     # 振动持续抬升(1x频为主)
            "temperature": (-99.0, 2.0, 0.3),  # 温度基本正常
            "current": (0.0, 2.0, 0.5),        # 电流轻度波动
        },
        "severity": "medium",
        "risk_of_failure_hours": 336,
        "symptoms": "振幅随转速平方增长, 1 倍频占主导, 风机叶轮常见积尘结垢所致",
        "actions": [
            "采集振动频谱, 确认 1x 转频分量占比",
            "停机检查叶轮积尘/结垢情况, 进行清灰除垢",
            "对转子做动平衡校正(现场动平衡仪)",
            "复查对中状态与地脚螺栓预紧力",
        ],
        "spare_parts": ["动平衡配重片若干"],
    },
    {
        "code": "F003",
        "name": "轴系不对中",
        "component": "联轴器",
        "applicable_devices": ["motor", "fan", "gearbox"],
        "signature": {
            "vibration": (2.0, 99.0, 2.0),     # 振动上升(2x频特征)
            "temperature": (0.5, 4.0, 1.0),    # 联轴器局部温升
            "current": (0.0, 2.5, 0.8),        # 电流波动(周期性负载变化)
        },
        "severity": "medium",
        "risk_of_failure_hours": 360,
        "symptoms": "轴向振动偏大, 2 倍频分量明显, 联轴器边缘温度高于本体",
        "actions": [
            "使用激光对中仪复查电机-负载轴系同轴度",
            "调整垫片完成冷态对中(径向/轴向偏差<0.05mm)",
            "检查联轴器弹性元件是否老化破损",
            "复紧地脚螺栓并做基础共振检查",
        ],
        "spare_parts": ["弹性联轴器膜片 x1", "不锈钢调整垫片若干"],
    },
    {
        "code": "F004",
        "name": "润滑不足/润滑失效",
        "component": "润滑系统",
        "applicable_devices": ["motor", "fan", "gearbox"],
        "signature": {
            "temperature": (2.5, 99.0, 3.0),   # 温度显著上升(第一信号)
            "vibration": (0.5, 5.0, 1.5),      # 振动中度上升(油膜不稳)
        },
        "severity": "high",
        "risk_of_failure_hours": 120,
        "symptoms": "油温/绕组温度快速爬升, 油样金属颗粒增多, 严重时啸叫",
        "actions": [
            "立即检查油位/油镜, 不足则补充指定牌号润滑油",
            "提取油样做铁谱分析, 判断磨损金属含量",
            "检查油路是否堵塞, 清洗或更换滤芯/油嘴",
            "润滑失效且温度继续上升时, 按规程停机检修",
        ],
        "spare_parts": ["齿轮油 L-CKC220 x20L", "滤芯 x2"],
    },
    {
        "code": "F005",
        "name": "绕组绝缘老化",
        "component": "电机绕组",
        "applicable_devices": ["motor"],
        "signature": {
            "temperature": (2.0, 99.0, 2.5),   # 绕组温升显著
            "current": (1.5, 99.0, 2.0),       # 电流异常上升(绝缘下降/局部短路)
            "vibration": (-99.0, 3.0, 0.3),
        },
        "severity": "critical",
        "risk_of_failure_hours": 72,
        "symptoms": "三相电流不平衡度增大, 绝缘电阻下降, 温升超限并有绝缘焦味",
        "actions": [
            "测量三相电流不平衡度与绝缘电阻(应>1MΩ/kV)",
            "做匝间耐压试验, 判断是否存在匝间短路",
            "加强通风散热, 降载 30% 运行直至检修",
            "绝缘失效风险高时立即停机, 安排绕组重绕或更换电机",
        ],
        "spare_parts": ["备用电机 45kW x1", "绝缘漆 x5kg"],
    },
    {
        "code": "F006",
        "name": "供电电压异常",
        "component": "供电回路",
        "applicable_devices": ["motor", "fan", "gearbox"],
        "signature": {
            "voltage": (-99.0, -2.0, 3.0),     # 电压显著偏低(欠压)
        },
        "severity": "medium",
        "risk_of_failure_hours": 240,
        "symptoms": "母线电压低于额定 95%, 电流被动增大, 长期欠压导致绕组过热",
        "actions": [
            "用万用表/电能质量分析仪测量三相电压与不平衡度",
            "检查进线柜熔断器、接触器触点是否氧化烧蚀",
            "核实变压器分接头位置, 必要时申请调压",
            "欠压超过额定 -10% 时按规程降载或停机",
        ],
        "spare_parts": ["交流接触器 x1", "熔断器 x3"],
    },
    {
        "code": "F007",
        "name": "机械过载",
        "component": "传动系统",
        "applicable_devices": ["motor", "fan", "gearbox"],
        "signature": {
            "current": (2.5, 99.0, 3.0),       # 电流显著超限(第一信号)
            "temperature": (1.0, 6.0, 1.5),    # 温度随之上升
            "vibration": (0.0, 4.0, 0.8),
        },
        "severity": "medium",
        "risk_of_failure_hours": 200,
        "symptoms": "电流超过额定值且持续, 电机温升加快, 多因工艺负荷突增或卡滞",
        "actions": [
            "核对工艺负荷, 排除物料卡滞/管道堵塞",
            "检查传动机构(皮带张紧、联轴器)是否存在额外阻力",
            "临时降载至额定 85% 以下运行",
            "持续过载则按规程停机排查机械阻力源",
        ],
        "spare_parts": [],
    },
    {
        "code": "F008",
        "name": "齿轮点蚀/断齿",
        "component": "齿轮副",
        "applicable_devices": ["gearbox"],
        "signature": {
            "vibration": (3.0, 99.0, 3.0),     # 啮合频率冲击显著
            "temperature": (1.5, 99.0, 1.5),   # 油温上升
            "current": (0.5, 3.0, 0.8),
        },
        "severity": "high",
        "risk_of_failure_hours": 144,
        "symptoms": "振动频谱出现齿轮啮合频率及其边频带, 油中铁磁颗粒浓度上升",
        "actions": [
            "采集振动频谱, 分析啮合频率边频带特征",
            "开箱检查齿面点蚀/剥落位置与面积占比",
            "做油液铁谱分析, 量化磨损趋势",
            "点蚀面积超标或断齿风险时更换齿轮副, 并同步更换润滑油",
        ],
        "spare_parts": ["齿轮副 x1", "骨架油封 x2"],
    },
]


# ------------------------------------------------------------------------------
# 故障分类器
# ------------------------------------------------------------------------------
class FaultClassifier:
    """基于特征签名匹配的故障分类器。"""

    def __init__(self, library=None, top_k: int = 3):
        self.library = library if library is not None else FAULT_LIBRARY
        self.top_k = top_k

    def classify(self, z_scores: dict, device_type: str,
                 health_score: float = None) -> dict:
        """对一组 z 分数特征做故障匹配。

        Args:
            z_scores:     {通道: z分数}, 来自预测引擎的标准化偏差
            device_type:  设备类型(motor/fan/gearbox), 用于过滤适用故障
            health_score: 可选健康评分, 用于提升紧急度判断

        Returns:
            {matched: [...], ruled_out: 布尔, summary: 文本}
            matched 按置信度降序, 每项含 code/name/confidence/reason。
        """
        candidates = []
        for fault in self.library:
            if device_type and device_type not in fault["applicable_devices"]:
                continue
            confidence, evidence = self._score_fault(fault, z_scores)
            candidates.append({
                "code": fault["code"],
                "name": fault["name"],
                "component": fault["component"],
                "severity": fault["severity"],
                "confidence": round(confidence, 3),
                "evidence": evidence,
                "risk_of_failure_hours": fault["risk_of_failure_hours"],
                "symptoms": fault["symptoms"],
                "actions": fault["actions"],
                "spare_parts": fault["spare_parts"],
            })
        # 按置信度排序取 Top-K
        candidates.sort(key=lambda c: c["confidence"], reverse=True)
        matched = candidates[:self.top_k]
        # 只有第一名置信度达到阈值才算"有效命中", 否则视为证据不足
        ruled_out = not matched or matched[0]["confidence"] < 0.35
        summary = self._summarize(matched, ruled_out, health_score)
        return {"matched": matched, "ruled_out": ruled_out, "summary": summary}

    def _score_fault(self, fault: dict, z_scores: dict) -> tuple:
        """计算单个故障的置信度: 分级证据得分 / 全部签名权重之和。

        证据强度为连续分级(0~1), 避免"二值命中"导致多故障同时饱和:
          - 开口区间(单边恶化): 从区间下界起, 偏离越深证据越足,
            达到下界的 2 倍(或 +1σ)记满分;
          - 有界区间(上下限均有限, 表示"该通道应处于此范围"):
            落入即满分, 超出无证据;
          - 方向相反(签名要求升温而实测显著降温)按 30% 权重扣分。
        """
        total_weight = 0.0
        score = 0.0
        evidence = []
        for channel, (lo, hi, weight) in fault["signature"].items():
            total_weight += weight
            z = z_scores.get(channel)
            if z is None:
                continue
            strength = self._evidence_strength(z, lo, hi)
            if strength > 0.0:
                score += weight * strength
                evidence.append("%s 偏离 %+.1fσ 命中特征区间(证据强度 %.0f%%)"
                                % (channel, z, strength * 100))
            elif self._contradicts(z, lo, hi):
                # 方向相反: 该证据反对本故障
                score -= 0.3 * weight
                evidence.append("%s 偏离 %+.1fσ 与特征方向不符" % (channel, z))
        confidence = max(0.0, score / total_weight) if total_weight > 0 else 0.0
        return confidence, evidence

    @staticmethod
    def _evidence_strength(z: float, lo: float, hi: float) -> float:
        """计算单通道证据强度(0~1), 分开口区间与有界区间两种情形。"""
        if lo > 0 or (lo == 0 and hi >= 50):
            # 上开口: 证据从 z=lo 开始爬坡, 至 z=max(lo*2, lo+1) 达到满分
            full = max(lo * 2.0, lo + 1.0)
            return clip01((z - lo) / (full - lo))
        if hi < 0 and lo <= -50:
            # 下开口(如欠压): 从 z=hi 向下爬坡, 至 z=min(hi*2, hi-1) 满分
            full = min(hi * 2.0, hi - 1.0)
            return clip01((hi - z) / (hi - full))
        # 有界区间: 落入即满分
        return 1.0 if lo <= z <= hi else 0.0

    @staticmethod
    def _contradicts(z: float, lo: float, hi: float) -> bool:
        """判定实测方向是否与签名期望方向相反(显著才判反对)。"""
        if lo > 0 and z < lo - 1.5:
            return True          # 期望上升通道反而显著低于基线
        if hi < 0 and z > hi + 1.5:
            return True          # 期望下降通道反而显著高于基线
        return False

    def _summarize(self, matched: list, ruled_out: bool, health_score) -> str:
        """生成诊断结论文本。"""
        if ruled_out:
            return ("当前异常证据不足以定位具体故障, 建议加密监测频率(缩短至 5 分钟/次), "
                    "重点关注振动频谱与温升速率变化。")
        top = matched[0]
        lines = ["最可能故障: %s(%s, 置信度 %.0f%%)"
                 % (top["name"], top["code"], top["confidence"] * 100)]
        if len(matched) > 1:
            others = ", ".join("%s(%.0f%%)" % (m["name"], m["confidence"] * 100)
                               for m in matched[1:] if m["confidence"] > 0.15)
            if others:
                lines.append("次要可能: " + others)
        if health_score is not None:
            lines.append("当前健康评分 %.1f, 建议在 %d 小时内完成处置。"
                         % (health_score, top["risk_of_failure_hours"]))
        return " ".join(lines)

    def recommend_actions(self, matched: list) -> list:
        """汇总 Top 故障的处置方案(去重保序), 供告警引擎生成工单。"""
        actions = []
        for m in matched:
            if m["confidence"] < 0.35:
                continue
            actions.extend(m["actions"])
        seen, unique = set(), []
        for a in actions:
            if a not in seen:
                seen.add(a)
                unique.append(a)
        return unique or ["加密监测, 持续观察 24 小时"]

    def find_by_code(self, code: str) -> dict:
        """按故障编码查询知识库条目(告警引擎关联推荐时使用)。"""
        for fault in self.library:
            if fault["code"] == code:
                return fault
        return None


# ------------------------------------------------------------------------------
# 内置自检用例: 构造典型故障特征, 验证分类正确性
# ------------------------------------------------------------------------------
SELF_TESTS = [
    {
        "name": "轴承磨损用例(晚期冲击特征)",
        "device_type": "motor",
        "z_scores": {"vibration": 7.0, "temperature": 4.5, "current": 1.5, "voltage": 0.1},
        "expect_top": "F001",
    },
    {
        "name": "润滑不足用例(温升主导)",
        "device_type": "gearbox",
        "z_scores": {"vibration": 1.5, "temperature": 4.5, "current": 0.6, "voltage": 0.0},
        "expect_top": "F004",
    },
    {
        "name": "供电欠压用例",
        "device_type": "fan",
        "z_scores": {"vibration": 0.5, "temperature": 0.8, "current": 1.8, "voltage": -4.0},
        "expect_top": "F006",
    },
    {
        "name": "过载用例(电流主导)",
        "device_type": "motor",
        "z_scores": {"vibration": 1.2, "temperature": 2.6, "current": 4.5, "voltage": -0.4},
        "expect_top": "F007",
    },
]


def run_selftest() -> bool:
    """运行内置用例, 打印匹配结果并返回是否全部通过。"""
    clf = FaultClassifier()
    all_pass = True
    for case in SELF_TESTS:
        result = clf.classify(case["z_scores"], case["device_type"], health_score=70)
        top = result["matched"][0] if result["matched"] else None
        ok = top is not None and top["code"] == case["expect_top"]
        all_pass = all_pass and ok
        status = "PASS" if ok else "FAIL"
        print("[%s] %s -> 命中 %s (期望 %s)"
              % (status, case["name"],
                 top["code"] + top["name"] if top else "无",
                 case["expect_top"]))
        print("       %s" % result["summary"])
    print("\n自检结果: %s" % ("全部通过" if all_pass else "存在失败用例"))
    return all_pass


def main():
    parser = argparse.ArgumentParser(description="数字孪生平台 - 故障分类器")
    parser.add_argument("--selftest", action="store_true", help="运行内置用例自检")
    parser.add_argument("--library", action="store_true", help="打印故障知识库")
    args = parser.parse_args()
    if args.library:
        print(json.dumps(FAULT_LIBRARY, ensure_ascii=False, indent=2))
    else:
        run_selftest()


if __name__ == "__main__":
    main()
