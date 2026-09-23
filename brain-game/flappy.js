// FlyBrain Flappy — Laya 决策 → 果蝇左/右腿动作 → 上/下按钮 → 小鸟飞行
// ---------------------------------------------------------------------------
// 一个页面三块：
//   · 左上：Flappy 小鸟游戏（UP/DOWN 按钮控制小鸟升降）
//   · 左下：真·果蝇 3D 模型（MuJoCo/WASM），做对应的左/右腿蹬动作
//   · 右侧：Laya 决策 + LIF 神经放电可视化
//
// 闭环：把小鸟状态编成英文句 → Laya 决策 FLAP/HOLD → 果蝇左腿(FLAP)或右腿(HOLD)
// 蹬动（3D 里真的能看到腿在动）→ 果蝇的动作点亮 UP/DOWN 按钮 → 按钮驱动小鸟。
// 果蝇是这只小鸟的"驾驶员"，它的腿就是在按按钮。

import * as THREE from 'three';
import { loadScene, buildMeshes, syncMeshes, makeStepper } from '../shared/scene.js';
import { Controller } from '../shared/controller.js';

const PIPELINE_URL = 'http://localhost:8000';
const AI_INTERVAL_MS = 120;              // Laya 决策间隔（后端实测~15ms，完全撑得住）
const ASSETS = './assets';

// 小鸟画布 / 物理
const W = 380, H = 440;
const GRAVITY = 620;                     // px/s^2（调柔，下坠不那么猛）
const FLAP_IMPULSE = -255;               // 一记向上脉冲的速度 (px/s)，调弱避免上冲过猛
const PIPE_W = 58, GAP = 185, PIPE_SPEED = 100, PIPE_INTERVAL = 2.4;
const GROUND_H = 50;

// 13 运动神经元标签 (对齐 motor_map.py)
const NEURON_NAMES = ['wDL','wEL','wDR','wER','hL','hR','lFL','lML','lHL','lFR','lMR','lHR','CMD'];
const WING = [0, 1, 2, 3];

// 3D 果蝇腿：leg_order = [lf, lm, lh, rf, rm, rh] → 前3左、后3右
const LEFT_LEGS = [0, 1, 2];
const RIGHT_LEGS = [3, 4, 5];

const canvas = document.getElementById('game');
const ctx = canvas.getContext('2d');

const game = {
  phase: 'ready',            // ready | running | dead
  birdY: H / 2, birdV: 0,
  pipes: [], spawnT: 0,
  score: 0, best: 0,
  wingPhase: 0,
};

const ai = {
  choice: '—', confidence: 0, state: '', latency: 0,
  raster: [], rates: new Array(13).fill(0),
  wingAmp: 0, legL: 0, legR: 0,
  busy: false, lastCall: 0,
  pressUp: false,            // 果蝇现在在"按"上还是下
  spikeCount: 0, activeCount: 0,
  history: [],               // 决策流水
};

// --- 把游戏状态编成 Laya 能懂的英文句子（二元决策） -----------------------
// 实测这两句能让 Laya 稳定二分：太低→FLAP(conf~0.86)，下降静止→HOLD(conf~0.55)。
// 数值状态串会把置信度压到 <0.1，所以必须用自然语言情境句。
const STATE_UP   = 'The fly is flying too low and must flap its wings hard to rise up.';
const STATE_DOWN = 'The fly is descending and resting its wings, staying still and calm.';

function buildState() {
  const gap = nextGap();
  const target = gap ? gap.top + GAP * 0.55 : H / 2;   // 矄准缝隙略偏下，补偿上冲惯性
  const predicted = game.birdY + game.birdV * 0.16;    // 多看一点下坠速度，提前收手
  return predicted > target ? STATE_UP : STATE_DOWN;
}

function nextGap() {
  for (const p of game.pipes) {
    if (p.x + PIPE_W > 70) return { top: p.gapTop, x: p.x };
  }
  return null;
}

// --- 调 Laya /decide 拿回决策 + 神经放电 ---------------------------------
async function decide(now) {
  if (ai.busy || now - ai.lastCall < AI_INTERVAL_MS) return;
  ai.busy = true; ai.lastCall = now;
  const state = buildState();
  ai.state = state;
  try {
    const t0 = performance.now();
    const res = await fetch(`${PIPELINE_URL}/decide`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ state, neural_steps: 40, dry_run: true }),
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const j = await res.json();
    ai.latency = Math.round(performance.now() - t0);
    applyDecision(j);
  } catch (e) {
    console.warn('[flappy] /decide failed:', e);
  }
  ai.busy = false;
}

// --- 决策 → 神经放电 → 果蝇左/右腿 → 上/下 → 小鸟 ------------------------
function applyDecision(j) {
  ai.choice = j.laya?.choice || '?';
  ai.confidence = j.laya?.confidence || 0;
  ai.raster = j.neural?.raster || [];
  ai.rates = j.neural?.firing_rates || ai.rates;
  ai.latency = j.total_latency_ms || ai.latency;

  const r = ai.rates;
  ai.wingAmp = (r[WING[0]] + r[WING[1]] + r[WING[2]] + r[WING[3]]) / 4;

  // 总放电数（raster 里所有 spike 之和）——衡量这次“思考”有多活跃
  ai.spikeCount = ai.raster.reduce((s, row) => s + row.reduce((a, v) => a + v, 0), 0);
  ai.activeCount = r.filter((v) => v > 0.3).length;

  // 二元：FLAP → 蹬左腿 = 按 UP（小鸟上升）；HOLD → 蹬右腿 = 松/按 DOWN（重力下落）
  ai.pressUp = ai.choice === 'FLAP';
  ai.legL = ai.pressUp ? 1 : 0;
  ai.legR = ai.pressUp ? 0 : 1;

  // 记一笔决策流水（最新在前，保留 12 条）
  ai.history.unshift({ choice: ai.choice, conf: ai.confidence, up: ai.pressUp });
  if (ai.history.length > 12) ai.history.pop();

  // 命令 3D 果蝇：让对应侧的三条腿走一个步态周期
  if (fly) fly.triggerLegs(ai.pressUp ? 'left' : 'right');

  // 给小鸟一记向上脉冲（等于果蝇的腿按下了 UP 键）
  if (ai.pressUp && game.phase === 'running') game.birdV = FLAP_IMPULSE;
}

// --- 小鸟物理 -------------------------------------------------------------
function update(dt) {
  game.wingPhase += dt * (8 + ai.wingAmp * 40);
  if (game.phase !== 'running') return;

  game.birdV += GRAVITY * dt;
  game.birdY += game.birdV * dt;

  game.spawnT += dt;
  if (game.spawnT >= PIPE_INTERVAL) {
    game.spawnT = 0;
    const margin = 55;
    const gapTop = margin + Math.random() * (H - GROUND_H - GAP - margin * 2);
    game.pipes.push({ x: W, gapTop, scored: false });
  }
  for (const p of game.pipes) {
    p.x -= PIPE_SPEED * dt;
    if (!p.scored && p.x + PIPE_W < 70) { p.scored = true; game.score++; }
  }
  game.pipes = game.pipes.filter((p) => p.x + PIPE_W > -10);

  if (game.birdY > H - GROUND_H - 10 || game.birdY < 10) return die();
  for (const p of game.pipes) {
    if (70 + 12 > p.x && 70 - 12 < p.x + PIPE_W) {
      if (game.birdY - 10 < p.gapTop || game.birdY + 10 > p.gapTop + GAP) return die();
    }
  }
}

function die() { game.phase = 'dead'; game.best = Math.max(game.best, game.score); }

function reset() {
  game.phase = 'running';
  game.birdY = H / 2; game.birdV = 0;
  game.pipes = []; game.spawnT = PIPE_INTERVAL - 0.4;
  game.score = 0;
}

// --- 小鸟渲染 -------------------------------------------------------------
function render() {
  ctx.clearRect(0, 0, W, H);
  for (const p of game.pipes) {
    const g = ctx.createLinearGradient(p.x, 0, p.x + PIPE_W, 0);
    g.addColorStop(0, '#4a8f3c'); g.addColorStop(.5, '#6fc255'); g.addColorStop(1, '#4a8f3c');
    ctx.fillStyle = g;
    ctx.fillRect(p.x, 0, PIPE_W, p.gapTop);
    ctx.fillRect(p.x, p.gapTop + GAP, PIPE_W, H - GROUND_H - p.gapTop - GAP);
    ctx.fillStyle = '#3f7a33';
    ctx.fillRect(p.x - 4, p.gapTop - 14, PIPE_W + 8, 14);
    ctx.fillRect(p.x - 4, p.gapTop + GAP, PIPE_W + 8, 14);
  }
  ctx.fillStyle = '#2a1f10'; ctx.fillRect(0, H - GROUND_H, W, GROUND_H);
  ctx.fillStyle = '#3a2c16';
  for (let i = 0; i < W; i += 22) ctx.fillRect(i, H - GROUND_H, 11, 5);

  drawBird(70, game.birdY);

  ctx.fillStyle = '#fff'; ctx.font = '800 34px -apple-system,sans-serif';
  ctx.textAlign = 'center'; ctx.shadowColor = 'rgba(0,0,0,.6)'; ctx.shadowBlur = 6;
  ctx.fillText(game.score, W / 2, 56); ctx.shadowBlur = 0;

  if (game.phase === 'ready') banner('AI 准备就绪', '空格 / 点击开始 · Laya 驾驶果蝇');
  else if (game.phase === 'dead') banner('撞上了 💥', `得分 ${game.score} · 最佳 ${game.best} · 空格重来`);
}

function banner(title, sub) {
  ctx.fillStyle = 'rgba(0,0,0,.55)'; ctx.fillRect(0, H / 2 - 55, W, 110);
  ctx.fillStyle = '#fff'; ctx.textAlign = 'center';
  ctx.font = '800 24px -apple-system,sans-serif'; ctx.fillText(title, W / 2, H / 2 - 8);
  ctx.font = '500 12px -apple-system,sans-serif'; ctx.fillStyle = '#cfd3d8';
  ctx.fillText(sub, W / 2, H / 2 + 20);
}

// 小鸟角色（简笔黄色小鸟，翅膀随神经放电扑动）
function drawBird(x, y) {
  const tilt = Math.max(-0.5, Math.min(0.7, game.birdV / 600));
  ctx.save(); ctx.translate(x, y); ctx.rotate(tilt);
  const wf = Math.sin(game.wingPhase) * (0.5 + ai.wingAmp * 0.5);
  ctx.fillStyle = '#e8b84a';
  ctx.beginPath(); ctx.ellipse(0, 0, 15, 12, 0, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = '#f5d27a';                       // 翅膀
  ctx.beginPath(); ctx.ellipse(-3, 2, 9, 6 + wf * 5, -0.4, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = '#fff';                          // 眼
  ctx.beginPath(); ctx.arc(8, -4, 4, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = '#222'; ctx.beginPath(); ctx.arc(9, -4, 2, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = '#e8853a';                       // 喙
  ctx.beginPath(); ctx.moveTo(14, -2); ctx.lineTo(22, 0); ctx.lineTo(14, 3); ctx.fill();
  ctx.restore();
}

// --- 神经 HUD -------------------------------------------------------------
function updateHud() {
  dom.choice.textContent = ai.choice;
  dom.conf.textContent = `置信度 ${(ai.confidence * 100).toFixed(0)}%`;
  dom.reason.textContent = reasonText();
  dom.state.textContent = '“' + (ai.state || '…') + '”';
  dom.latency.textContent = `延迟 ${ai.latency} ms · 每 ${AI_INTERVAL_MS}ms 决策一次`;
  dom.btnUp.classList.toggle('on', ai.pressUp);
  dom.btnDown.classList.toggle('on', !ai.pressUp);
  dom.legL.style.width = (ai.legL * 100).toFixed(0) + '%';
  dom.legR.style.width = (ai.legR * 100).toFixed(0) + '%';
  dom.wing.style.width = (ai.wingAmp * 100).toFixed(0) + '%';
  dom.spikeStats.innerHTML =
    `网络 <b>${13 + SATELLITES.length}</b> 节点 · 总放电 <b>${ai.spikeCount}</b> · 活跃核心 <b>${ai.activeCount}</b>/13`;
  dom.rates.innerHTML = ai.rates.map((r, i) =>
    `<span class="${r > 0.3 ? 'on' : ''}">${NEURON_NAMES[i]}</span>`).join('');
  dom.history.innerHTML = ai.history.map((h) => {
    const cls = h.up ? 'up' : 'down';
    const pct = (h.conf * 100).toFixed(0);
    return `<div class="hrow ${cls}"><span class="tag">${h.choice}</span>` +
           `<span class="hbar"><i style="width:${pct}%"></i></span>` +
           `<span class="pct">${pct}%</span></div>`;
  }).join('');
  drawBrainMap();
}

// 把当前决策翻译成人话，让面板看起来在“思考”
function reasonText() {
  if (ai.choice === 'FLAP')
    return '→ 感知到小鸟偏低，命令神经元驱动左侧翅/腿 → 按 UP';
  if (ai.choice === 'HOLD')
    return '→ 小鸟偏高，收束翅膀靠重力下降 → 松/DOWN';
  return '→ 等待下一个决策…';
}

// 神经元"连接组"迷你脑图：把 13 个运动神经元按解剖位置摆成一张小图。
// 坐标归一化 (0~1)，按果蝇神经解剖：翅膀在上、haltere 在胸部两侧、
// 六条腿分左右两列、CMD 命令神经元在中枢。
const NODES = [
  { x: 0.30, y: 0.12 }, // 0 wDL 左翅
  { x: 0.40, y: 0.16 }, // 1 wEL
  { x: 0.70, y: 0.12 }, // 2 wDR 右翅
  { x: 0.60, y: 0.16 }, // 3 wER
  { x: 0.18, y: 0.34 }, // 4 hL  左 haltere
  { x: 0.82, y: 0.34 }, // 5 hR  右 haltere
  { x: 0.24, y: 0.55 }, // 6 lFL 左前腿
  { x: 0.20, y: 0.74 }, // 7 lML 左中腿
  { x: 0.26, y: 0.92 }, // 8 lHL 左后腿
  { x: 0.76, y: 0.55 }, // 9 lFR 右前腿
  { x: 0.80, y: 0.74 }, // 10 lMR 右中腿
  { x: 0.74, y: 0.92 }, // 11 lHR 右后腿
  { x: 0.50, y: 0.50 }, // 12 CMD 中枢命令
];
// CMD 向所有运动神经元投射（简化的连接组）
const EDGES = [0,1,2,3,4,5,6,7,8,9,10,11].map((n) => [12, n]);
function nodeHue(n) { return n < 4 ? 45 : n < 6 ? 200 : n < 12 ? 130 : 285; }

// --- 卡群中间神经元：围绕每个核心运动神经元生成一群“卫星”节点。
// 它们的放电跟随对应核心神经元（带噪声/相位延迟），只是可视化。
// 核心决策逻辑完全不变——真正干活的还是那 13 个神经元。
const SATS_PER_CORE = 22;
const SATELLITES = [];
(function buildSatellites() {
  let seed = 1337;
  const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
  for (let core = 0; core < NODES.length; core++) {
    const spread = core === 12 ? 0.20 : 0.11;    // CMD 中枢撑得更开
    const count = core === 12 ? SATS_PER_CORE * 2 : SATS_PER_CORE;
    for (let k = 0; k < count; k++) {
      const ang = rnd() * Math.PI * 2;
      const r = Math.sqrt(rnd()) * spread;         // 均匀铺在圆盘内
      SATELLITES.push({
        core,
        x: Math.min(0.98, Math.max(0.02, NODES[core].x + Math.cos(ang) * r)),
        y: Math.min(0.98, Math.max(0.02, NODES[core].y + Math.sin(ang) * r * 0.9)),
        phase: rnd() * Math.PI * 2,               // 闪烁相位
        thr: 0.15 + rnd() * 0.5,                   // 跟随阈值（噪声）
        rad: 0.8 + rnd() * 1.4,
      });
    }
  }
})();

// 最近一步的瞬时放电（用来让节点闪一下）
function latestSpikes() {
  const r = ai.raster;
  return r.length ? r[r.length - 1] : null;
}

function drawBrainMap() {
  const cw = dom.raster.width, ch = dom.raster.height;
  rctx.clearRect(0, 0, cw, ch);
  rctx.fillStyle = '#0e1017'; rctx.fillRect(0, 0, cw, ch);
  const px = (nx) => 10 + nx * (cw - 20);
  const py = (ny) => 8 + ny * (ch - 16);
  const rates = ai.rates, spikes = latestSpikes();
  const now = performance.now() / 1000;

  // 卫星中间神经元：先画它们到核心的细线 + 节点，形成密集脑图背景。
  // 每个卫星活跃度 = 对应核心放电率过阈 + 噪声闪烁。
  for (const s of SATELLITES) {
    const core = s.core;
    const drive = rates[core] || 0;
    // 基础微光（即使核心安静也有微弱背景活动）+ 驱动亮度
    const idle = 0.06;
    const flick = 0.55 + 0.45 * Math.sin(now * 7 + s.phase);
    const driven = drive > s.thr ? (drive - s.thr) / (1 - s.thr) : 0;
    const act = (idle + driven) * flick;
    if (act <= 0.02) continue;
    const sx = px(s.x), sy = py(s.y);
    const cx = px(NODES[core].x), cy = py(NODES[core].y);
    const hue = nodeHue(core);
    // 到核心的细纤维
    rctx.strokeStyle = `hsla(${hue},75%,60%,${0.05 + act * 0.28})`;
    rctx.lineWidth = 0.5;
    rctx.beginPath(); rctx.moveTo(sx, sy); rctx.lineTo(cx, cy); rctx.stroke();
    // 卫星节点
    rctx.fillStyle = `hsla(${hue},85%,${45 + act * 35}%,${0.35 + act * 0.6})`;
    rctx.beginPath(); rctx.arc(sx, sy, s.rad, 0, Math.PI * 2); rctx.fill();
  }

  // 连线：基础暗线，目标神经元放电时亮起来（命令传导的观感）
  for (const [a, b] of EDGES) {
    const act = rates[b] || 0;
    rctx.strokeStyle = `hsla(${nodeHue(b)},80%,60%,${0.08 + act * 0.5})`;
    rctx.lineWidth = 0.6 + act * 1.6;
    rctx.beginPath();
    rctx.moveTo(px(NODES[a].x), py(NODES[a].y));
    rctx.lineTo(px(NODES[b].x), py(NODES[b].y));
    rctx.stroke();
  }

  // 节点：半径/亮度随放电率；刚发了 spike 的额外白边闪一下
  for (let n = 0; n < NODES.length; n++) {
    const x = px(NODES[n].x), y = py(NODES[n].y);
    const act = rates[n] || 0;
    const rad = n === 12 ? 6 : 4;
    const hue = nodeHue(n);
    // 光晕
    if (act > 0.05) {
      const g = rctx.createRadialGradient(x, y, 0, x, y, rad + 8);
      g.addColorStop(0, `hsla(${hue},90%,65%,${0.5 * act})`);
      g.addColorStop(1, 'hsla(0,0%,0%,0)');
      rctx.fillStyle = g;
      rctx.beginPath(); rctx.arc(x, y, rad + 8, 0, Math.PI * 2); rctx.fill();
    }
    // 节点主体
    const lit = 30 + act * 45;
    rctx.fillStyle = `hsl(${hue},85%,${lit}%)`;
    rctx.beginPath(); rctx.arc(x, y, rad, 0, Math.PI * 2); rctx.fill();
    // spike 白闪
    if (spikes && spikes[n]) {
      rctx.strokeStyle = 'rgba(255,255,255,.9)'; rctx.lineWidth = 1.4;
      rctx.beginPath(); rctx.arc(x, y, rad + 2, 0, Math.PI * 2); rctx.stroke();
    }
  }
}

// --- 3D 果蝇（真·MuJoCo 模型）-------------------------------------------
// 复用走路 demo 的 Controller（stepSingle：每条腿一个触发位）。收到 Laya 决策后，
// 让左侧或右侧三条腿走一个步态周期，视觉上就是"左脚蹬"或"右脚蹬"。
const MAX_SUBSTEPS = 60;
const PLAYBACK_SPEED = 0.12;          // 慢放，MuJoCo 在浏览器远低于实时
let fly = null;

class Fly3D {
  constructor(mj, model, data, meta) {
    this.mj = mj; this.model = model; this.data = data; this.meta = meta;
    this.dt = meta.timestep;
    this.controller = new Controller(meta);
    this.action = new Float64Array(6);      // 每条腿的触发位
    this.stepper = makeStepper(this.dt, MAX_SUBSTEPS);
    this.bodyId = this._flyBody();
    this._resetSim();
    this._buildScene();
  }

  _flyBody() {
    for (let j = 0; j < this.model.njnt; j++)
      if (this.model.jnt_type[j] === 0) return this.model.jnt_bodyid[j];
    return 1;
  }

  _resetSim() {
    this.mj.mj_resetDataKeyframe(this.model, this.data, 0);
    this.controller.reset();
    this.mj.mj_forward(this.model, this.data);
  }

  // Laya 决策来了：只让选中侧的三条腿走步态，另一侧清零（停止触发）。
  triggerLegs(side) {
    const on = side === 'left' ? LEFT_LEGS : RIGHT_LEGS;
    const off = side === 'left' ? RIGHT_LEGS : LEFT_LEGS;
    for (const i of on) this.action[i] = 1;    // stepSingle 检测>0 启动一个步态周期
    for (const i of off) this.action[i] = 0;
  }

  _buildScene() {
    const stage = document.getElementById('stage');
    THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    stage.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.9));
    const key = new THREE.DirectionalLight(0xffffff, 1.1); key.position.set(6, -8, 14);
    const fill = new THREE.DirectionalLight(0xffffff, 0.4); fill.position.set(-6, 5, 5);
    this.scene.add(key, fill);

    // 地面棋盘
    const geo = new THREE.PlaneGeometry(60, 60);
    const mat = new THREE.MeshStandardMaterial({ color: 0x8ace00, roughness: 0.95 });
    const ground = new THREE.Mesh(geo, mat); ground.position.z = -0.02;
    this.scene.add(ground);

    this.meshGroup = buildMeshes(this.model, this.meta);
    this.scene.add(this.meshGroup);

    this.camera = new THREE.PerspectiveCamera(50, 1, 0.05, 200);
    this.camera.up.set(0, 0, 1);
    addEventListener('resize', () => this._resize());
    this._resize();
    syncMeshes(this.meshGroup, this.data);
    this._updateCamera();
  }

  _resize() {
    const el = document.getElementById('stage-wrap');
    const w = el.clientWidth, h = el.clientHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / Math.max(1, h);
    this.camera.updateProjectionMatrix();
  }

  // 固定俯视斜角，始终对准果蝇（不追朝向，保持稳定好观察腿）
  _updateCamera() {
    const b = this.bodyId, d = this.data;
    const fx = d.xpos[3 * b], fy = d.xpos[3 * b + 1], fz = d.xpos[3 * b + 2];
    this.camera.position.set(fx - 3.2, fy - 3.2, fz + 3.0);
    this.camera.lookAt(fx, fy, fz + 0.2);
  }

  _physicsStep() {
    const ctrl = this.data.ctrl;
    this.controller.stepSingle(ctrl, this.action);
    this.mj.mj_step(this.model, this.data);
  }

  frame(wallDt) {
    this.stepper.advance(wallDt * PLAYBACK_SPEED, () => { this._physicsStep(); });
    syncMeshes(this.meshGroup, this.data);
    this._updateCamera();
    this.renderer.render(this.scene, this.camera);
    // 腿指示灯：任意左/右腿正在步态中就点亮
    const lActive = LEFT_LEGS.some((i) => this.controller.legPhases[i] > 0);
    const rActive = RIGHT_LEGS.some((i) => this.controller.legPhases[i] > 0);
    dom.pipLeft.classList.toggle('active', lActive);
    dom.pipRight.classList.toggle('active', rActive);
  }
}

// --- DOM 引用 -------------------------------------------------------------
const dom = {
  choice: document.getElementById('ai-choice'),
  conf: document.getElementById('ai-conf'),
  reason: document.getElementById('ai-reason'),
  state: document.getElementById('ai-state'),
  latency: document.getElementById('ai-latency'),
  rates: document.getElementById('rates'),
  spikeStats: document.getElementById('spike-stats'),
  history: document.getElementById('history'),
  raster: document.getElementById('raster'),
  btnUp: document.getElementById('btn-up'),
  btnDown: document.getElementById('btn-down'),
  legL: document.getElementById('leg-l'),
  legR: document.getElementById('leg-r'),
  wing: document.getElementById('wing-amp'),
  pipLeft: document.getElementById('pip-left'),
  pipRight: document.getElementById('pip-right'),
  overlay: document.getElementById('stage-overlay'),
};
const rctx = dom.raster.getContext('2d');

// --- 主循环 ---------------------------------------------------------------
let lastT = 0;
function loop(nowMs) {
  requestAnimationFrame(loop);
  const now = nowMs / 1000;
  const dt = lastT ? Math.min(now - lastT, 0.05) : 0;
  lastT = now;

  if (game.phase === 'running') decide(nowMs);
  update(dt);
  render();
  updateHud();
  if (fly) fly.frame(dt);
}

// --- 输入 -----------------------------------------------------------------
function startOrRestart() {
  if (game.phase === 'ready' || game.phase === 'dead') reset();
}
addEventListener('keydown', (e) => {
  if (e.code === 'Space') { e.preventDefault(); startOrRestart(); }
  if (e.code === 'ArrowUp' && game.phase === 'running') game.birdV = FLAP_IMPULSE;
});
canvas.addEventListener('pointerdown', startOrRestart);

// --- 启动：先加载 3D 果蝇，再开跑主循环 ----------------------------------
async function boot() {
  try {
    const { mj, model, data, meta } = await loadScene({
      assetsDir: ASSETS, xmlName: 'fly.xml',
      onStage: (msg) => { dom.overlay.textContent = msg; },
    });
    fly = new Fly3D(mj, model, data, meta);
    dom.overlay.classList.add('hidden');
  } catch (e) {
    console.error('[flappy] 3D fly failed to load:', e);
    dom.overlay.textContent = '3D 果蝇加载失败（游戏仍可玩）：' + e.message;
  }
  requestAnimationFrame(loop);
}
boot();
