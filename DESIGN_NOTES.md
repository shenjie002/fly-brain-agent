# FlyBrain Agent — 设计笔记

## 目标

构建一个 Web 闭环 Demo：Laya 快脑决策 → 神经放电(LIF) → 果蝇 3D 运动。
果蝇模型来自 FlyGym (NeuroMechFly v2)，物理由浏览器内 MuJoCo WASM 驱动。

目前包含两个前端页面，共用同一套后端与 3D 果蝇：
- **game.html** — 自由赛道，果蝇用 CPG 六足步态自主穿越
- **flappy.html** — FlyBrain Flappy，果蝇的运动神经"驾驶"一只 Flappy 小鸟

## 架构

```
浏览器 (brain-game/)                     Python 后端 (pipeline.py :8000)
┌──────────────────────┐                ┌──────────────────────┐
│ MuJoCo WASM 物理     │  POST /decide  │ Laya 快脑 (laya-mlx) │
│ Three.js 果蝇渲染    │ ◄────────────► │ LIF 神经放电 (motor_map)│
│ CPG 六足控制器       │                │ 运动指令解码          │
│ AI HUD (神经面板)    │                └──────────────────────┘
└──────────────────────┘
```

## 关键模块

### 1. Laya 快脑 (Python)
- 包: laya-mlx (convaiinnovations/laya, ModernBERT)
- 输入: 英文自然语言状态描述 (中文 conf<0.1，不可用)
- 输出: choice (FLAP/HOLD/WALK/TURN_LEFT/TURN_RIGHT) + confidence
- 延迟: 首次 ~3s (加载), 后续 ~15ms

### 2. 神经放电 LIF (Python, motor_map.py)
- 13 个运动神经元: 4 wing + 2 haltere + 6 leg + 1 CMD
- choice 决定哪些运动池被驱动; confidence 只调制强度 (0.65+0.35*conf)
- 输出: spike raster (40步×13), firing_rates, MotorCommand

**LIF 是什么 (Leaky Integrate-and-Fire, 泄漏积分-发放)**
最经典的脉冲神经元模型。把神经元看成一个会漏水的杯子:
- **Integrate(积分)**: 输入让"膜电位"(水位)上涨, 初值 -65mV
- **Leak(泄漏)**: 没有新输入时水位按 `decay=exp(-dt/tau)` 慢慢漏回静息值,
  防止久远输入无限累积 (simulator.py: `v = v*decay + R*(I+synaptic)`)
- **Fire(发放)**: 水位越过阈值 -50mV 就发一个 spike(输出1), 随即清零重来
- 神经元之间由连接矩阵 W 相连: 一个发放会经突触 `W@spikes_prev`
  把电流传给下游, 影响它们的水位 → 构成网络

**在本项目里 LIF 干嘛**: 它是 Laya"想法"和果蝇"动作"之间的翻译层——
让"决策→动作"真的走一遍神经动力学, 而不是一个 if-else。链路:
```
Laya choice → 13 神经元输入电流 → LIF 跑 40 步(积分-泄漏-发放)
  → 40×13 spike raster → 数发放次数得 firing_rate
  → 解码成翅膀振幅 / 左右腿增益等运动指令
```
前端脑图上闪烁的节点, 就是这 13 个 LIF 神经元在积分-发放。

### 3. 运动指令 → CPG 控制 (JS, brain-game/game.js)
- flygym 原生 Controller.stepCPG(ctrl, gainL, gainR)
- Laya 决策映射: WALK→gainL=gainR=1, TURN_LEFT→gainL=0.4/gainR=1.2, 等
- 真 MuJoCo WASM 物理驱动六足

### 4. 3D 果蝇渲染 (JS, flygym scene.js)
- STL 部件由 MuJoCo 加载, buildMeshes() 构建 Three.js 网格
- syncMeshes() 每帧从仿真数据读位姿 → 写入网格矩阵
- 需要 COOP/COEP 头 (SharedArrayBuffer 依赖)

## 文件结构

```
fly-brain-agent/
├── pipeline.py          # 端到端 FastAPI 服务 (Laya + LIF + 运动解码)
├── motor_map.py         # 13 神经元 LIF 网络 + 运动指令解码
├── build_manifest.py    # [已废弃] 自烘焙几何导出 (有 bug, 不再使用)
├── serve.py             # 本地 dev server: COOP/COEP 头 + /decide 反代到 :8000
├── brain-game/          # 两个前端页面, 共用后端 + 3D 果蝇
│   ├── game.html        # [页面1] 自由赛道 + 神经放电 HUD 面板
│   ├── game.js          # AIBrain(fetch /decide) + Game._aiTick → stepCPG()
│   ├── flappy.html      # [页面2] FlyBrain Flappy: 神经驾驶小鸟 + Neural Cockpit
│   ├── flappy.js        # 游戏物理 + Laya 决策循环 + 连接组脑图可视化
│   └── assets/          # fly.xml + STL + model_meta.json
├── shared/              # 前端共用层
│   ├── scene.js         # flygym 3D 场景 (STL 加载 + Three.js 网格同步)
│   ├── controller.js    # 从 game.js 抽出的 CPG 六足步态控制器
│   └── vendor/          # three.js + mujoco wasm
├── web/                 # [旧] 自定义 viewer (模型有问题, 已切换到 brain-game)
│   ├── viewer.html
│   ├── viewer.js
│   └── model/           # 自烘焙 .bin 文件 (废弃)
└── DESIGN_NOTES.md      # 本文件
```

## 已解决的关键问题

1. **Laya 是英文模型** — 中文 state 置信度全 <0.1; 换英文后决策正确
2. **驱动曾 ∝ 绝对置信度** — 低 conf 时动作退化; 改成 choice 主导
3. **自烘焙几何碎片** — build_manifest.py 丢了 mesh_face 面索引;
   最终决定: 不自己烘焙, 直接用 flygym 原生 STL + MuJoCo 加载

## 已完成 (闭环)

- [x] brain-game/game.js 注入 Laya→LIF→CPG 控制
      - 新增 'AI' 关卡 (🧠 Laya Brain 按钮)。Game._aiTick() 每 800ms
        POST /decide, 用 choice→CPG 双侧增益 map 驱动 stepCPG():
        WALK/FLAP=对称, TURN_LEFT=0.4/1.2, TURN_RIGHT=1.2/0.4, HOLD=0/0;
        增益强度按 0.65+0.35*conf 调制 (choice 主导, conf 只调幅)。
- [x] AI 状态 + 神经放电 HUD 面板
      - #neural-hud: choice + confidence + 延迟, 13 神经元 label 高亮,
        raster canvas (wing=黄 / haltere=蓝 / leg=绿 / CMD=紫)。
- [x] 端到端浏览器验证闭环
      - serve.py (COOP/COEP + wasm mime + /decide 反代) 全绿:
        game.html/scene.js/vendor/assets 均 200, wasm=application/wasm。
        模拟浏览器决策循环: 状态→choice 正确, warmup 后 ~13ms/次。

## FlyBrain Flappy (flappy.html / flappy.js)

第二个前端页面: 用果蝇的运动神经"驾驶"一只 Flappy 小鸟。一屏三块:
- **左上 · Flappy 小鸟** — 不是人玩, 是 Laya 在开; ↑UP/↓DOWN 按钮实时高亮
- **左下 · 3D 果蝇** — 真 MuJoCo 模型, FLAP 蹬左腿 / HOLD 蹬右腿
- **右侧 · Neural Cockpit** — 决策/置信度/推理文字 + 连接组脑图 + 放电统计 + 决策流水

完整闭环:
```
小鸟位置 → 编成英文情境句 → Laya 决策 (FLAP/HOLD)
  → 13 个 LIF 神经元放电 → 果蝇左/右腿蹬动
  → 点亮 UP/DOWN 按钮 → 升力脉冲/重力下落 → 小鸟飞行
```

**调参 (让神经驾驶员能玩)**: 后端 /decide 实测 ~15ms, 不是瓶颈。
- 决策间隔 380ms → 120ms (快 3 倍, 反应不再迟钝)
- 重力 1100 → 620, 拍翅脉冲 -255 (单次上冲 ~52px, 缝隙 185px)
- 瞄准点 GAP*0.55 + 预测提前量 0.16, 补偿上冲惯性, 上下平衡

**连接组脑图 (drawBrainMap)**: 底层仍是 13 个真实运动神经元做决策;
每个核心神经元周围散布 ~22 个装饰性"卫星"中间神经元 (CMD 44 个),
总计 ~335 节点, 放电跟随所属核心(带噪声+微光闪烁)。这些卫星是
可视化增强, 不参与决策 —— 要真正上百个有功能神经元需换后端模型。

**诚实边界**: FLAP 只驱动翅膀、HOLD 全静默, 左右腿放电率本身区分不了
上下; 所以 UP/DOWN 是用 Laya 的 choice 直接映射左/右腿的。

## 关键点: state 必须是自然英语

Laya 是英文语言模型, 数值 state 串 ("x=1.2, speed=2.4...") 置信度塌到
<0.1。_buildStateStr() 改成场景化英语句子 ("The fly has drifted to the
left of the path, it should turn right..."), choice 正确且 conf 0.4~0.6。

## 运行方式

1. 起后端: `.venv/bin/uvicorn pipeline:app` (或 python pipeline.py) → :8000
2. 起前端: `python serve.py` → http://localhost:8090/brain-game/
3. 两个页面:
   - `game.html` — 点 🧠 AI 关卡, 果蝇由 Laya 快脑自主穿越赛道
   - `flappy.html` — 果蝇运动神经驾驶 Flappy 小鸟, 右侧看神经驾驶舱
