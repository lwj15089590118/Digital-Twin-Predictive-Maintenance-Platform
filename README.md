# Digital-Twin-Predictive-Maintenance-Platform

[![CI](https://github.com/lwj15089590118/Digital-Twin-Predictive-Maintenance-Platform/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/lwj15089590118/Digital-Twin-Predictive-Maintenance-Platform/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> 数字孪生驱动的设备智能运维平台 —— 让设备"开口说话"，把故障消灭在发生之前。

## 一、项目简介

本项目是一套面向工业设备（主电机、离心风机、齿轮传动箱）的**数字孪生 + AI 预测性维护**一体化演示平台。它以"物理设备实时数据"为输入，在信息空间中构建并持续校准设备的数字镜像，再由 AI 引擎对数字孪生的状态进行异常检测、健康评分与剩余寿命（RUL）预测，最终驱动告警引擎自动生成工单与处置方案，完成"**事后维修 → 事前预防**"的运维模式升级。

平台内置完整的数据闭环：**数据模拟器**可以仿真设备从健康到故障的全生命周期衰退过程（振动上升、温升加大、电流增加、转速下降），因此无需真实产线即可端到端体验预测性维护的全部能力；接入真实设备时，只需将模拟器替换为 OPC-UA / MQTT 采集程序，其余模块零改动。

## 二、闭环架构

```mermaid
flowchart LR
    subgraph P1[1. 物理设备层]
        A1[主电机 MOTOR-001]
        A2[离心风机 FAN-001]
        A3[齿轮传动箱 GEARBOX-001]
    end

    subgraph P2[2. 数据采集层]
        B1[振动/温度/电流/电压传感器<br/>10分钟采样]
        B2[data_simulator 数据模拟器<br/>含设备衰退趋势仿真]
    end

    subgraph P3[3. 数字孪生层]
        C1[model_updater 模型更新器<br/>物理→数字 在线校准]
        C2[孪生状态库<br/>基线漂移/偏差记录/同步度]
        C3[数字→物理 反向控制建议]
    end

    subgraph P4[4. AI 预测层]
        D1[predictive_engine 预测引擎<br/>异常检测 IsolationForest/Mahalanobis]
        D2[健康评分 0~100%<br/>多通道加权融合]
        D3[RUL 剩余寿命预测<br/>趋势外推+80%置信区间]
        D4[fault_classifier 故障分类器<br/>故障知识库匹配 8 类故障]
    end

    subgraph P5[5. 运维决策层]
        E1[alert_engine 告警引擎<br/>分级告警/抑制/升级]
        E2[维修工单自动生成<br/>关联处理方案与备件]
        E3[dashboard 综合看板<br/>3D车间+曲线+报告导出]
    end

    A1 & A2 & A3 --> B1 --> B2
    B2 -- 实时数据流 JSONL --> C1
    C1 --> C2
    C2 -- 偏差/漂移特征 --> D1
    D1 --> D2 --> D3
    D2 & D3 & D4 --> E1
    D1 -- 异常模式 z分数 --> D4
    E1 --> E2 --> E3
    C3 -- 降载/巡检建议 --> P1
    E3 -. 运维决策反馈 .-> P1
```

数据闭环的三条关键链路：

1. **感知链路**：物理设备 → 传感器 → 数据模拟器 → 实时数据流（`data/stream/`）；
2. **孪生链路**：实时数据 → 数字孪生在线校准（EWMA 吸收基线漂移）→ 偏差记录与同步度评分 → AI 预测引擎；
3. **决策链路**：健康评分 / RUL / 故障归因 → 分级告警与工单 → 3D 看板可视化 → 反向控制建议回馈物理设备。

## 三、核心功能

| 功能 | 说明 |
| --- | --- |
| 数据模拟 | 3 台设备全生命周期衰退仿真，含 5 种故障模式注入（轴承磨损、转子不平衡、润滑不足、齿轮点蚀、绕组绝缘老化） |
| 数字孪生 | 物理-数字双向映射：在线校准基线参数、记录通道级偏差、输出同步度与反向控制建议 |
| 异常检测 | sklearn 可用时使用 IsolationForest，否则自动降级为 Mahalanobis 距离，任何环境可运行 |
| 健康评分 | 振动/温度/电流/电压/转速五通道加权融合，输出 0~100% |
| RUL 预测 | 健康评分退化趋势外推 + 拟合残差构造 80% 置信区间 |
| 故障诊断 | 8 类故障知识库签名匹配，输出置信度、处置方案与备件清单 |
| 告警工单 | 提醒/严重/紧急三级告警 + RUL 预测预警，严重/紧急级别自动生成工单并关联方案 |
| 3D 可视化 | Three.js 数字车间，设备颜色随健康评分绿→黄→橙→红渐变，点击查看详情 |
| 综合看板 | 实时曲线、健康趋势、设备总览、告警列表、预测报告（Markdown）一键导出 |

## 四、目录结构

```
Digital-Twin-Predictive-Maintenance-Platform/
├── README.md                        # 本文件
├── LICENSE                          # MIT 许可证
├── .github/workflows/ci.yml         # CI: 语法检查 + 3 个自检(compileall/selftest)
├── data-ingestion/
│   ├── data_simulator.py            # 数据模拟器(实时流 + 历史数据集生成)
│   └── historical_data.csv          # 历史训练数据集(686条, 含正常/故障前数据)
├── digital-twin/
│   └── model_updater.py             # 数字孪生模型更新器(双向映射/偏差记录)
├── ai-prediction/
│   ├── predictive_engine.py         # AI预测引擎(异常检测/健康评分/RUL)
│   ├── fault_classifier.py          # 故障分类器(8类故障知识库)
│   └── evaluate.py                  # 定量评估(混淆矩阵/RUL误差/CI覆盖率)
├── reports/                         # 评估工件(evaluation_report.md / .json)
├── visualization/
│   └── 3d_scene/app.js              # Three.js 3D数字车间
├── dashboard/
│   └── app.py                       # Flask综合看板(集成全部能力)
├── alerts/
│   └── alert_engine.py              # 告警引擎(分级告警/工单生成)
├── docs/
│   ├── 系统设计说明书.md             # 完整设计文档(架构/算法选型/集成方案)
│   └── img/                          # README 截图(评估数据 matplotlib 渲染)
├── resume/
│   └── 项目总结.md                   # 第一人称项目总结
└── data/                            # 运行时目录(数据流/孪生状态/告警, 自动生成)
```

## 五、快速开始

### 环境要求

- Python 3.10+，依赖见 [`requirements.txt`](requirements.txt)：`flask`、`numpy`（必需）；`scikit-learn`（可选，缺失时自动降级为纯统计实现）
- 一键安装：`pip install -r requirements.txt`（如需启用 IsolationForest/随机森林，再安装 `scikit-learn`）
- 现代浏览器（3D 场景需要 WebGL 支持，Three.js 与 Chart.js 通过 CDN 加载，首次打开需联网）

### 一键启动（推荐）

```bash
python dashboard/app.py
# 访问 http://127.0.0.1:5000
# 看板启动时会自动生成历史数据集并训练模型, 约 5~10 秒后出现数据
```

看板后台自动驱动完整闭环流水线：模拟器 → 数字孪生 → AI 预测 → 告警工单。演示节奏下设备约 7~9 分钟走完一个生命周期（每 2 秒推进 6 个模拟循环，三台设备寿命 1320~1560 循环），可点击"复位演示"重新观察。

### 分模块运行

```bash
# 1. 单独生成历史训练数据集
python data-ingestion/data_simulator.py --mode csv --rows-per-device 220

# 2. 训练预测引擎并查看训练摘要
python ai-prediction/predictive_engine.py --train

# 3. 故障分类器/告警引擎自检
python ai-prediction/fault_classifier.py --selftest
python alerts/alert_engine.py --selftest

# 4. AI 基线定量评估(混淆矩阵/RUL 误差/CI 覆盖率, 工件写入 reports/)
python ai-prediction/evaluate.py
python ai-prediction/evaluate.py --selftest   # RUL 置信区间单元用例(快/慢退化)

# 5. 数字孪生单步演示(快进900循环, 观察偏差演化)
python digital-twin/model_updater.py --once

# 6. 实时数据流模式(供孪生跟随消费)
python data-ingestion/data_simulator.py --mode stream --interval 1
python digital-twin/model_updater.py --follow
```

## 六、关键设计决策

1. **零依赖降级**：所有 AI 组件在 sklearn 缺失时自动切换到 numpy 统计实现（Mahalanobis 异常检测、规则阶段分类、最小二乘趋势外推），保证演示环境开箱即用；
2. **规则 + 数据双引擎诊断**：故障知识库的物理签名匹配提供可解释归因，随机森林提供统计判别，两者互补；
3. **数字孪生可解释同步**：每次同步输出通道级"实测 vs 模型"残差与同步度评分，偏差即早期故障特征来源；
4. **预测驱动告警**：RUL 低于 72 小时或健康评分跌破阈值时触发分级告警，其中健康评分跌破严重/紧急阈值时自动生成工单（RUL 预测预警仅提示、不生成工单），处置方案与备件清单直接来自故障库，形成管理闭环。

更多架构细节、算法选型论证（LSTM vs 随机森林）与真实产线集成方案，请阅读 [`docs/系统设计说明书.md`](docs/系统设计说明书.md)。

## 七、AI 基线定量评估

本项目是规则/统计基线（健康评分 + 阶段分类 + RUL 趋势外推），未训练深度模型；评估脚本 [`ai-prediction/evaluate.py`](ai-prediction/evaluate.py) 在与训练集（seed 2026）无样本重叠的独立仿真轨迹（seed 7，三台设备全生命周期 4695 样本）上做留出评估，工件提交于 [`reports/evaluation_report.md`](reports/evaluation_report.md)。核心实测数字（numpy-fallback 后端）：

| 指标 | 实测值 |
| --- | --- |
| 阶段分类准确率 / 宏 F1 | 92.3% / 0.890 |
| 黄金窗口（真值健康度 0.6~0.8）漏检率 | 25.8%（修复前死区设计为 40.3%） |
| 黄金窗口 RUL 不可估占比 | 9.8%（修复前 35.1%） |
| RUL 中位相对误差 / MAE | 88.4% / 296.4 h（中位绝对误差 40.3 h） |
| 80% 置信区间实测覆盖率 | 33.0%（残差自相关使独立性假设低估长程不确定性，叠加退化前半程凸性导致的线性外推系统性高估，如实报告，未达名义 80%） |

> RUL 口径说明（v1.3 已统一）：评估与看板流水线均按每循环 1 条记录喂入（1 条 = 1 tick = 10 分钟），真值与预测/置信区间统一为小时，共用 RULPredictor 同一换算——同一轨迹（种子 7）上看板在线 RUL 与评估口径实测逐位一致（可比点 mismatch=0），预测预警（72h 窗口）主体触发点真值 RUL 62~89h、与名义窗口对齐（v1.2 及之前看板每 6 个循环仅把最后 1 条喂给引擎，在线 RUL 失真至约 1/3~1/6，预警在真值 RUL≈432h 即提前触发，本轮已修复）。冷启动前约 40 tick 评分平坦段存在拟合噪声误触发，评估路径同样存在（同源），随运行历史积累消失。早期退化段线性外推存在单点最高约 68 倍的高估，该基线适用于趋势参考与回归对比，不应作为精确检修时刻依据。
>
> 局限声明：特征与标签同源于自仿真数据（同一健康度真值同时决定传感器读数与标签），以上指标系统性偏高，仅用于回归对比与缺陷验证，不代表真实产线精度。复现方式见评估报告第 4 节。

## 八、项目实况截图（评估数据渲染）

以下两张图均由 [`ai-prediction/evaluate.py`](ai-prediction/evaluate.py) 的留出评估数据（seed 7，与 [`reports/evaluation_report.md`](reports/evaluation_report.md) 同一轨迹、同一引擎，本轮复跑指标逐项一致：准确率 92.3%、黄金窗口漏检 25.8%、RUL 覆盖率 33.0%）经 matplotlib 渲染，未做任何手工修饰。

![健康度退化曲线](docs/img/health_degradation.png)

*三台设备全生命周期健康度退化曲线：灰色为真值健康度（×100），彩色为平台校准健康评分；黄色带为 60~80 黄金维护窗口，虚线为 35 分失效阈值。校准评分全程跟踪真值，黄金窗口内评分平均绝对偏差 5.40 分。*

![RUL 预测 vs 真值](docs/img/rul_pred_vs_truth.png)

*RUL 预测 vs 真值散点（主口径，可估样本 n=2424，对数坐标）：图内标注均为实测值——80% CI 实测覆盖率 33.0%、MAE 296.4h、中位绝对误差 40.3h、近失效窗口中位 15.3h；106 个评分提前跌破阈值输出 0.0h 的样本未绘制。早期凸性高估与晚期代理评分噪声如实呈现。*

## 九、FAQ

**Q1：AI 是真模型还是规则？**
如实说：本项目是**规则/统计基线**，未训练深度模型。异常检测在 scikit-learn 可用时使用 IsolationForest（训练式），缺失时自动降级为 Mahalanobis 距离 + 规则引擎（numpy-fallback，两者输出接口完全一致）；阶段分类在 sklearn 路径为随机森林、缺失时为规则判别（评估工件即在后一后端实测）；健康评分经保序回归校准表（17 节点分段线性）映射；深度方案（LSTM）的取舍论证见 [`docs/系统设计说明书.md`](docs/系统设计说明书.md)。README §七 与评估报告对该定位有明确声明。

**Q2：RUL 是怎么算的？**
对最近 144 条（= 一整天，覆盖温度日周期完整周期，避免日周期伪斜率污染斜率估计）健康评分做最小二乘线性外推至 35 分失效阈值，按 1 tick = 10 分钟换算为小时；80% 置信区间由拟合残差的标准外推预测方差构造：√(1+1/n+(k-x̄)²/Sxx)·σ ÷ |b| × TICK_HOURS（margin 先除以斜率绝对值再换算量纲，修复过程见复审报告 07-P1-1）。评估真值口径为"真值健康度首次跌破 0.35 的剩余时长"，经唯一换算点 `truth_rul_hours()` 统一为小时。

**Q3：置信区间为什么实测覆盖率（33.0%）低于名义 80%？**
如实归因，未修饰：其一，拟合残差存在自相关，"残差独立"假设低估了长程不确定性；其二，健康度按幂函数先缓后急退化，线性外推在退化前半程系统性高估（主口径单点最高 68.4 倍）。因此该 RUL 基线定位为**趋势参考与回归对比**，不应作为精确检修时刻依据（docs §15 已列为已知边界）。

**Q4：没装 scikit-learn 能跑吗？降级路径是什么？**
能。全链路 numpy-fallback：Mahalanobis 距离替代 IsolationForest、规则阶段分类、最小二乘趋势外推，输出接口与 sklearn 路径完全一致；[`requirements.txt`](requirements.txt) 中 scikit-learn 默认注释为可选，README §七 的全部评估数字即在 numpy-fallback 后端实测生成。

**Q5：健康分怎么校准？会不会偷看真值？**
单通道恶化度（z>0.5σ 起算，按灵敏度分配满量程跨度：振动 42σ、温度/电流 20σ、双边 8σ）加权融合后，经 17 节点保序回归（isotonic 分段线性）校准表映射，使校准评分 ≈ 100×真值健康度。校准表与跨度常数在 4 条独立仿真轨迹（seed 2026/11/22/33，约 1.9 万样本）上离线标定，与评估种子 7 不相交（无泄漏，复审报告 07 已核查）；**运行时推理不读取真值 health 字段**，等价于工业现场用失效数据离线标定评分模型。

**Q6：告警与工单什么时候自动生成？**
健康评分跌破提醒（75）/严重（55）/紧急（35）阈值触发三级告警；其中跌破**严重/紧急**阈值时自动生成维修工单（RUL 预测预警仅提示、不生成工单），处置方案与备件清单来自 8 类故障知识库签名匹配。告警引擎经 RLock 覆盖评估/关单/查询/复位/落盘五条路径，JSON 原子落盘（tmp + os.replace）。

## 十、Roadmap（复审报告 07 残留项）

复审 07 行动项 #1（RUL 真值量纲修正 + 工件重生成）、#2（看板/评估/预警喂入口径统一）已在 `57cc438` / `316e027` 完成，#6 的"最小 CI"由本轮 `.github/workflows/ci.yml` 完成；以下为剩余真实规划项：

- [ ] `Pipeline.reset()` 收敛到流水线线程执行（队列/标志位），或为 `engine.health_history` 与 TwinModel 加与 DataStore 同款锁（复审07 #3）
- [ ] AlertEngine 查询接口返回 deepcopy、内存告警/工单截断 `[-500:]`、告警升级时收敛旧工单（复审07 #4）
- [ ] 补校准表构建脚本 `scripts/fit_calibration.py` 入库，使 docs §6.3 标定过程可从仓库复现（复审07 #5）
- [ ] pytest 包装三个 selftest，把"修复前后回归对比"固化为测试断言（复审07 #6 剩余部分）
- [ ] `TwinUpdater.save` 改原子写 + `demo_follow` 对未登记 device_id 跳过并告警（复审07 #7）
- [ ] 包结构化（pyproject/`__init__.py`）消除 4 处 sys.path hack；3D 电机轴加键槽标记（复审07 #8）
