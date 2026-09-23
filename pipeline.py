"""
pipeline.py — 端到端闭环: Laya决策 → 神经放电渲染 → 运动输出
================================================================
链路（用户定义）:
  [1] Laya 快脑: 读果蝇当前状态 -> choice(FLAP/HOLD/...) + confidence
  [2] fly-brain-agent 神经项目: 根据 Laya 决策激活运动神经元池,
      LIF 网络跑放电 -> 渲染神经元放电序列
  [3] 神经放电 -> 解码成运动指令 (翅膀振幅/频率, 六足步态)
  [4] 运动指令 -> FlyGym NeuroMechFly (翅膀/六足动作)

用法:
  python pipeline.py            # 跑一轮 demo (FLAP + HOLD + WALK)
  python pipeline.py --serve    # FastAPI 模式, 暴露 /decide 给前端
"""
from __future__ import annotations

import os
import sys
import time
import json
import warnings

import numpy as np

# ── 神经 + 运动解码（本项目） ───────────────────────────────────────────── #
import motor_map
from motor_map import decode_spikes, laya_to_motor, PRIMITIVES


# ── Laya 快脑（laya-mlx 包） ────────────────────────────────────────────── #
# 懒加载：只在真正调用时 import（权重下载只在第一次）
_LAYA_AGENT = None


def get_laya():
    global _LAYA_AGENT
    if _LAYA_AGENT is None:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from laya_mlx import load
        warnings.filterwarnings("ignore", ".*temperatures.*", RuntimeWarning)
        print("[pipeline] loading Laya (convaiinnovations/laya) ...")
        _LAYA_AGENT = load("convaiinnovations/laya")
        print("[pipeline] Laya ready.")
    return _LAYA_AGENT


def laya_decide(state_str: str) -> dict:
    """调用 Laya：游戏状态 -> {choice, confidence, probabilities}"""
    agent = get_laya()
    questions = {
        "action": {
            "type": "choice",
            "instructions": (
                "A fruit fly needs to navigate gaps. Given the current flight "
                "state, decide the best motor action. FLAP = power wing stroke "
                "to gain altitude; HOLD = glide / hold position; WALK = land "
                "and walk; TURN_LEFT / TURN_RIGHT = yaw turn."
            ),
            "criteria": list(PRIMITIVES),
        }
    }
    out = agent.predict(state_str, questions)
    ans = out["answers"]["action"]
    return {
        "choice": ans["choice"],
        "confidence": ans["confidence"],
        "probabilities": ans.get("probabilities", {}),
        "raw": out,
    }


def state_to_str(**kw) -> str:
    """把物理状态格式化成 Laya 能读的自然语言串。"""
    parts = []
    for k, v in kw.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:+.2f}")
        else:
            parts.append(f"{k}={v}")
    return "Flight state: " + ", ".join(parts) + "."


# ── 渲染神经放电（终端 ASCII） ──────────────────────────────────────────── #
NEURON_LABELS = [
    "wing-DL", "wing-EL", "wing-DR", "wing-ER",
    "haltere-L", "haltere-R",
    "leg-FL", "leg-ML", "leg-HL", "leg-FR", "leg-MR", "leg-HR",
    "CMD",
]


def render_raster(raster: np.ndarray, width: int = 60) -> str:
    """把 (steps x N) spike raster 画成终端 ASCII 图。"""
    T, N = raster.shape
    if N != len(NEURON_LABELS):
        labels = [f"n{i}" for i in range(N)]
    else:
        labels = NEURON_LABELS
    # down-sample time to `width` columns
    step_per_col = max(1, T // width)
    lines = []
    for i in range(N):
        row = raster[:, i]
        chars = []
        for c in range(width):
            seg = row[c * step_per_col:(c + 1) * step_per_col]
            chars.append("█" if seg.sum() > 0 else "·")
        lines.append(f"{labels[i]:10s} {''.join(chars)}")
    header = " " * 11 + "time →"
    return "\n".join([header] + lines)


# ── FlyGym 运动执行（可选；离线时只打印） ──────────────────────────────── #
def drive_flygym(cmd: motor_map.MotorCommand, *, dry_run: bool = True) -> dict:
    """把运动指令送给 FlyGym NeuroMechFly。

    dry_run=True（默认）: 不加载 MuJoCo，只返回指令摘要（用于无 MuJoCo 环境验证）。
    dry_run=False: 真正实例化 NeuroMechFly，按 cmd 设置关节控制。
    """
    summary = {
        "flap_amp": round(cmd.flap_amp, 3),
        "flap_freq_hz": round(cmd.flap_freq, 3),
        "tripod_phase": round(cmd.tripod_phase, 3),
        "leg_gains": [round(float(g), 3) for g in cmd.leg_gains],
    }
    if dry_run:
        summary["flygym"] = "dry-run (no MuJoCo)"
        return summary

    # 真正驱动 FlyGym：需要把 flygym 作为正式依赖装进当前环境
    # （pip install flygym）。装了就用，没装就返回提示，不依赖任何外部目录。
    try:
        from flygym import NeuroMechFly  # noqa: F401
    except Exception as exc:
        summary["flygym"] = f"not-installed: {exc}"
        return summary

    fly = NeuroMechFly()
    summary["flygym"] = "loaded"
    # 这里把 cmd 映射到 fly 的 actuator；具体 mapping 取决于 FlyGym 版本
    # （演示阶段先占位，真正接 joint ctrl 在硬件确认后补）
    return summary


# ── 端到端一步 ──────────────────────────────────────────────────────────── #
def step_pipeline(state_str: str, *, neural_steps: int = 40,
                  dry_run: bool = True, verbose: bool = True) -> dict:
    """完整跑一步: Laya -> 神经 -> 运动指令 -> FlyGym"""
    t0 = time.perf_counter()

    # [1] Laya 决策
    dec = laya_decide(state_str)
    t1 = time.perf_counter()

    # [2] 神经放电（根据 Laya 决策激活运动神经元池）
    cmd, sim = laya_to_motor(dec["choice"], dec["confidence"], steps=neural_steps)
    t2 = time.perf_counter()

    # [3] 运动指令解码已在 cmd 里
    # [4] 驱动 FlyGym
    fly_summary = drive_flygym(cmd, dry_run=dry_run)
    t3 = time.perf_counter()

    result = {
        "state": state_str,
        "laya": {"choice": dec["choice"], "confidence": dec["confidence"],
                 "latency_ms": round((t1 - t0) * 1000, 1)},
        "neural": {
            "raster_shape": list(cmd.raw_spikes.shape) if cmd.raw_spikes is not None else None,
            "firing_rates": [round(float(r), 3) for r in cmd.firing_rates],
            "latency_ms": round((t2 - t1) * 1000, 1),
            # 放电序列 (steps x 13) 的 0/1 矩阵, 前端画神经元闪烁
            "raster": cmd.raw_spikes.astype(int).tolist() if cmd.raw_spikes is not None else [],
        },
        "motor": fly_summary,
        # 运动指令: 前端直接拿来驱动果蝇
        "motor_cmd": {
            "flap_amp": round(cmd.flap_amp, 3),
            "flap_freq": round(cmd.flap_freq, 3),
            "tripod_phase": round(cmd.tripod_phase, 3),
            "leg_gains": [round(float(g), 3) for g in cmd.leg_gains],
        },
        "total_latency_ms": round((t3 - t0) * 1000, 1),
    }

    if verbose:
        print("=" * 72)
        print(f"STATE   : {state_str}")
        print(f"LAYA    : {dec['choice']}  (conf={dec['confidence']:.3f}, "
              f"{result['laya']['latency_ms']}ms)")
        print(f"NEURAL  : {result['neural']['latency_ms']}ms  "
              f"rates={['%.2f' % r for r in cmd.firing_rates]}")
        print(render_raster(cmd.raw_spikes))
        print(f"MOTOR   : flap_amp={cmd.flap_amp:.2f}  flap_freq={cmd.flap_freq:.2f}Hz  "
              f"legs={[round(float(g), 2) for g in cmd.leg_gains]}")
        print(f"FLYGYM  : {fly_summary.get('flygym')}")
        print(f"TOTAL   : {result['total_latency_ms']}ms")
    return result


# ── FastAPI 服务模式 ────────────────────────────────────────────────────── #
try:
    from pydantic import BaseModel, Field

    class DecideRequest(BaseModel):
        state: str = Field(..., description="自然语言描述的果蝇飞行状态")
        dry_run: bool = True
        neural_steps: int = 40
except Exception:  # pydantic 不在环境里时不阻断 CLI
    DecideRequest = None


def build_app():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="FlyBrain Pipeline", version="0.1.0")
    # 允许前端 (localhost:8090 等) 跨域调用
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root(): return {"service": "flybrain-pipeline", "status": "ready"}

    @app.post("/decide")
    def decide(req: DecideRequest):
        """POST body: {"state": "...", "neural_steps": 40, "dry_run": true}"""
        return step_pipeline(req.state, neural_steps=req.neural_steps,
                             dry_run=req.dry_run, verbose=False)

    return app

app = build_app()


# ── CLI ─────────────────────────────────────────────────────────────────── #
def main():
    args = sys.argv[1:]
    if args and args[0] == "--serve":
        import uvicorn
        uvicorn.run("pipeline:app", host="0.0.0.0",
                    port=int(os.environ.get("PORT", "8000")), reload=False)
        return

    # demo: 跑五种典型场景。
    # 注意: Laya 是英文自然语言模型 —— 数值型 state 串置信度会崩到 <0.1
    # (见 DESIGN_NOTES)。所以这里用情境化的英文句子, 与 brain-game 前端
    # (_buildStateStr) 保持同一套措辞风格。
    scenarios = [
        ("FLAP 场景", "The fly is falling fast and about to hit the ground, "
                      "it must flap its wings hard to gain altitude."),
        ("HOLD 场景", "The fly is at a good height and gliding smoothly, "
                      "it should hold position and coast."),
        ("WALK 场景", "The fly has landed and is on the ground, "
                      "it should walk forward toward the goal."),
        ("TURN_LEFT 场景", "The fly needs to turn left to follow the path."),
        ("TURN_RIGHT 场景", "An obstacle is ahead on the left, "
                            "the fly should turn right to avoid it."),
    ]
    for label, st in scenarios:
        print("\n" + "#" * 72)
        print(f"# {label}")
        step_pipeline(st, neural_steps=40, dry_run=True)


if __name__ == "__main__":
    main()
