// neural-fallback.js
// ---------------------------------------------------------------------------
// 离线神经兜底：把 motor_map.py 的 13 神经元连接矩阵 + 刺激，和 simulator.py
// 的 LIF 动力学忠实移植到 JS。后端 /decide 不可达时（如 GitHub Pages 静态托管），
// 用这里在浏览器里算出同样的 spike raster / firing_rates，神经可视化照常工作。
//
// 唯一被替换的是 Laya 语言模型本身：离线时用关键词启发式从状态句推断
// FLAP/HOLD（其余神经动力学与后端逐位一致）。
// ---------------------------------------------------------------------------

// 神经元索引（对齐 motor_map.py）
const WING_DL = 0, WING_EL = 1, WING_DR = 2, WING_ER = 3;
const HALT_L = 4, HALT_R = 5;
const LEG_FL = 6, LEG_ML = 7, LEG_HL = 8, LEG_FR = 9, LEG_MR = 10, LEG_HR = 11;
const CMD = 12;
const N = 13;

// _offline_weights(): 手写 13x13 连接矩阵 (pre -> post)
function offlineWeights() {
  const W = Array.from({ length: N }, () => new Array(N).fill(0));
  // 翅膀动力/恢复：拮抗交替（中枢振荡器）
  W[WING_DL][WING_EL] = -1.2; W[WING_EL][WING_DL] = -1.2;
  W[WING_DR][WING_ER] = -1.2; W[WING_ER][WING_DR] = -1.2;
  // 左右翅交叉耦合，保持同相
  W[WING_DL][WING_DR] = 0.6; W[WING_DR][WING_DL] = 0.6;
  W[WING_EL][WING_ER] = 0.6; W[WING_ER][WING_EL] = 0.6;
  // haltere 耦合到翅膀运动（相位稳定器）
  W[HALT_L][WING_DL] = 0.5; W[HALT_R][WING_DR] = 0.5;
  // 六足 CPG：三角步态。tripodA = FL,MR,HL ; tripodB = FR,ML,HR
  const tripodA = [LEG_FL, LEG_MR, LEG_HL];
  const tripodB = [LEG_FR, LEG_ML, LEG_HR];
  for (const a of tripodA) for (const b of tripodB) { W[a][b] = -0.9; W[b][a] = -0.9; }
  for (const grp of [tripodA, tripodB])
    for (const i of grp) for (const j of grp) if (i !== j) W[i][j] = 0.7;
  // 下行命令神经元驱动一切
  for (const i of [WING_DL, WING_EL, WING_DR, WING_ER, HALT_L, HALT_R, ...tripodA, ...tripodB])
    W[CMD][i] = 0.4;
  return W;
}

// _offline_stimulus(): 给定 primitive 的外部注入电流 (长度 N)
function offlineStimulus(primitive, confidence) {
  const I = new Array(N).fill(0);
  const conf = Math.min(1, Math.max(0, confidence));
  const drive = 6.0 * (0.65 + 0.35 * conf);   // 3.9 .. 6.0
  I[CMD] = drive;
  if (primitive === 'FLAP') {
    I[WING_DL] = I[WING_EL] = I[WING_DR] = I[WING_ER] = drive * 1.8;
    I[HALT_L] = I[HALT_R] = drive * 0.9;
    // 飞行时抑制腿
    I[LEG_FL] = I[LEG_ML] = I[LEG_HL] = I[LEG_FR] = I[LEG_MR] = I[LEG_HR] = 0;
  } else if (primitive === 'WALK') {
    I[LEG_FL] = I[LEG_ML] = I[LEG_HL] = I[LEG_FR] = I[LEG_MR] = I[LEG_HR] = drive * 1.5;
    I[WING_DL] = I[WING_EL] = I[WING_DR] = I[WING_ER] = 0;
    I[HALT_L] = I[HALT_R] = 0;
  } else if (primitive === 'TURN_LEFT') {
    I[LEG_FL] = I[LEG_HL] = drive * 2.0;
    I[LEG_FR] = I[LEG_HR] = drive * 0.3;
    I[WING_DL] = I[WING_EL] = drive * 0.5;
  } else if (primitive === 'TURN_RIGHT') {
    I[LEG_FR] = I[LEG_HR] = drive * 2.0;
    I[LEG_FL] = I[LEG_HL] = drive * 0.3;
    I[WING_DR] = I[WING_ER] = drive * 0.5;
  }
  // HOLD: 只有 CMD 驱动 -> 翅膀滑翔、腿站立
  return I;
}

// LIFSimulator.run(): 忠实移植 simulator.py 的 LIF 动力学
// 参数对齐 build_lif(): dt=1, tau=6, threshold=-50, reset=-65
function runLIF(W, stim, steps, { dt = 1.0, tau = 6.0, threshold = -50.0, reset = -65.0, R = 1.0 } = {}) {
  const decay = Math.exp(-dt / tau);
  const v = new Array(N).fill(reset);
  let lastSpikes = new Array(N).fill(0);
  const raster = [];
  for (let t = 0; t < steps; t++) {
    // 突触电流：上一 step 的 spike 经 W 传导 (W @ lastSpikes)
    const synaptic = new Array(N).fill(0);
    for (let i = 0; i < N; i++) {
      let s = 0;
      for (let j = 0; j < N; j++) s += W[i][j] * lastSpikes[j];
      synaptic[i] = s;
    }
    const spikes = new Array(N).fill(0);
    for (let i = 0; i < N; i++) {
      v[i] = v[i] * decay + R * (stim[i] + synaptic[i]);
      if (v[i] >= threshold) { spikes[i] = 1; v[i] = reset; }
    }
    raster.push(spikes);
    lastSpikes = spikes;
  }
  return raster;   // (steps x N) of 0/1
}

// decode_spikes(): 从 raster 算 firing_rates（0..1）
function firingRates(raster) {
  const T = raster.length || 1;
  const rates = new Array(N).fill(0);
  for (const row of raster) for (let i = 0; i < N; i++) rates[i] += row[i];
  return rates.map((s) => s / T);
}

// 关键词启发式：离线时替代 Laya 语言模型，从英文状态句推断 FLAP/HOLD。
// 与前端 buildState() 产生的两句对齐（too low/flap -> FLAP，其余 -> HOLD）。
function heuristicChoice(state) {
  const s = (state || '').toLowerCase();
  const flap = /too low|flap|rise|climb|hard/.test(s);
  return { choice: flap ? 'FLAP' : 'HOLD', confidence: flap ? 0.86 : 0.55 };
}

// 顶层：状态句 -> 决策 + 神经放电，形状对齐后端 /decide 响应。
function decideOffline(state, steps = 40) {
  const { choice, confidence } = heuristicChoice(state);
  const W = offlineWeights();
  const stim = offlineStimulus(choice, confidence);
  const raster = runLIF(W, stim, steps);
  const rates = firingRates(raster);
  return {
    laya: { choice, confidence },
    neural: { raster, firing_rates: rates.map((r) => Math.round(r * 1000) / 1000) },
    total_latency_ms: 0,
    offline: true,
  };
}

export { decideOffline, offlineWeights, N };
