// ============================================================
//  FlyBrain Viewer — Laya 决策 → 神经放电 → 3D 果蝇动作
//  脑在 Python 后端 (pipeline.py /decide)，前端只渲染果蝇 + 神经
// ============================================================
import * as THREE from 'three';

const API = 'http://127.0.0.1:8000/decide';
const ASSET = './model/';
const $ = (id) => document.getElementById(id);
const clamp = (x, a, b) => x < a ? a : (x > b ? b : x);

// 13 个运动神经元标签（顺序须与 motor_map.py 一致）
const NRN = ['wing-DL','wing-EL','wing-DR','wing-ER','haltere-L','haltere-R',
             'leg-FL','leg-ML','leg-HL','leg-FR','leg-MR','leg-HR','CMD'];

// 每个场景 → 一段状态描述（喂给 Laya）。
// 注意: Laya 是英文 ModernBERT 模型，中文置信度极低，state 必须用英文。
const SCENES = {
  flap: 'Flight: altitude low and falling fast, the gap is above, must flap wings hard to climb.',
  hold: 'The fly is at the right height cruising level, do nothing, maintain current altitude, hold steady.',
  walk: 'On the ground now, an obstacle is ahead, walk forward on six legs.',
  turn: 'On the ground, the opening is to the left, need to turn left.',
};

// 六足铰接点 (身体坐标)，索引对应 leg_gains 顺序 FL ML HL FR MR HR
const LEG_BASE = [
  [0.35, -0.1, 1.2], [0.4, -0.1, 0.4], [0.4, -0.1, -0.4],
  [-0.35, -0.1, 1.2], [-0.4, -0.1, 0.4], [-0.4, -0.1, -0.4],
];

let renderer, scene, camera, fly, wingL, wingR;
let legNodes = [], neuronEls = [];
let curScene = 'flap';
let motor = { flap_amp: 0.5, flap_freq: 6, tripod_phase: 0, leg_gains: [0.5,0.5,0.5,0.5,0.5,0.5] };
let raster = [], rates = [], rasterStep = 0, rasterClock = 0, wingPhase = 0, lastT = 0, polling = false;

async function loadGeo(e) {
  const buf = await (await fetch(ASSET + e.file)).arrayBuffer();
  const dv = new DataView(buf);
  const n = dv.getInt32(0, true);
  const f = new Float32Array(buf, 4, n * 6);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(f.subarray(0, n * 3), 3));
  g.setAttribute('normal', new THREE.BufferAttribute(f.subarray(n * 3), 3));
  return g;
}
const matf = (rg) => new THREE.MeshStandardMaterial({
  color: new THREE.Color().setRGB(rg[0], rg[1], rg[2], THREE.SRGBColorSpace),
  roughness: 0.7, metalness: 0, side: THREE.DoubleSide,
  transparent: rg[3] < 1, opacity: rg[3], depthWrite: rg[3] >= 1,
});

init().catch((e) => { console.error(e); $('status').textContent = '加载失败: ' + e; });

async function init() {
  const mf = await (await fetch(ASSET + 'fly_manifest.json')).json();

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  $('stage').appendChild(renderer.domElement);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x10141c);
  camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.1, 120);
  camera.up.set(0, 1, 0);
  scene.add(new THREE.AmbientLight(0xffffff, 0.9));
  const key = new THREE.DirectionalLight(0xffffff, 1.2); key.position.set(-5, 9, 9); scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, 0.5); fill.position.set(6, -2, -7); scene.add(fill);

  fly = new THREE.Group(); fly.scale.setScalar(2.4);
  fly.add(new THREE.Mesh(await loadGeo(mf.body), matf(mf.body_rgba)));
  fly.add(new THREE.Mesh(await loadGeo(mf.eye), matf(mf.eye_rgba)));
  wingL = wingNode(await loadGeo(mf.wing_l), matf(mf.wing_rgba), mf.wing_l.root);
  wingR = wingNode(await loadGeo(mf.wing_r), matf(mf.wing_rgba), mf.wing_r.root);
  fly.add(wingL.group, wingR.group);
  buildLegs();
  scene.add(fly);

  camera.position.set(-9, 4, 15); camera.lookAt(0, 0, 1.5);
  buildNeuronPanel();
  bindUI();
  addEventListener('resize', onResize);
  renderer.setAnimationLoop(loop);
  $('status').textContent = '就绪 — 选择场景，向后端 Laya 请求决策';
  requestDecision();
}

function wingNode(geo, mat, base) {
  const group = new THREE.Group();
  const m = new THREE.Mesh(geo.clone(), mat);
  m.geometry.translate(-base[0], -base[1], -base[2]);
  group.add(m);
  group.position.set(base[0], base[1], base[2]);
  return { group };
}

function onResize() {
  renderer.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
}

function buildLegs() {
  legNodes = [];
  const mat = new THREE.MeshStandardMaterial({ color: 0x1a1206, roughness: 0.6 });
  for (let i = 0; i < 6; i++) {
    const hinge = new THREE.Group();
    hinge.position.set(...LEG_BASE[i]);
    const seg = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.03, 1.0, 8), mat);
    seg.position.y = -0.5;
    const side = i < 3 ? 1 : -1;
    hinge.rotation.z = side * 0.5;
    hinge.add(seg);
    fly.add(hinge);
    legNodes.push({ hinge, rest: side * 0.5, side });
  }
}

function buildNeuronPanel() {
  const box = $('neurons'); box.innerHTML = '';
  neuronEls = [];
  NRN.forEach((lab) => {
    const row = document.createElement('div'); row.className = 'nrn';
    const dot = document.createElement('span'); dot.className = 'dot';
    const name = document.createElement('span'); name.className = 'nname'; name.textContent = lab;
    row.append(dot, name);
    box.appendChild(row);
    neuronEls.push({ dot });
  });
}

function bindUI() {
  document.querySelectorAll('#scenarios button').forEach((b) => {
    b.onclick = () => {
      document.querySelectorAll('#scenarios button').forEach((x) => x.classList.remove('active'));
      b.classList.add('active');
      curScene = b.dataset.scene;
      requestDecision();
    };
  });
}

// 向后端请求一次决策 -> 更新 motor + raster + HUD
async function requestDecision() {
  if (polling) return;
  polling = true;
  $('status').textContent = '请求中… (Laya 推理)';
  try {
    const res = await fetch(API, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ state: SCENES[curScene], neural_steps: 40, dry_run: true }),
    });
    const d = await res.json();
    motor = d.motor_cmd || motor;
    raster = d.neural && d.neural.raster ? d.neural.raster : [];
    rates = (d.neural && d.neural.firing_rates) || [];
    rasterStep = 0;
    // 面板: 用放电率(0..1)设每个神经元点的稳定亮度, 不随帧乱闪
    neuronEls.forEach((el, i) => {
      const r = rates[i] != null ? rates[i] : 0;
      el.dot.style.background = r > 0.5 ? '#ff5a5a' : (r > 0.15 ? '#ffb14a' : '#2a3f5f');
      el.dot.style.boxShadow = r > 0.5 ? '0 0 8px #ff5a5a' : 'none';
      el.dot.style.opacity = (0.35 + 0.65 * r).toFixed(2);
    });
    $('act').textContent = d.laya.choice;
    $('confnum').textContent = d.laya.confidence.toFixed(3);
    $('conffill').style.width = (d.laya.confidence * 100) + '%';
    $('lat').textContent = 'Laya ' + d.laya.latency_ms + ' ms · 神经 ' + d.neural.latency_ms + ' ms';
    $('scene').textContent = curScene.toUpperCase();
    $('wingamp').textContent = (motor.flap_amp != null ? motor.flap_amp.toFixed(2) : '—');
    $('legphase').textContent = '[' + (motor.leg_gains || []).map((g) => g.toFixed(1)).join(' ') + ']';
    $('status').textContent = '决策完成 — 果蝇执行中';
    console.log('VIEWER_OK choice=' + d.laya.choice + ' conf=' + d.laya.confidence +
      ' flap_amp=' + motor.flap_amp + ' legs=' + JSON.stringify(motor.leg_gains) +
      ' raster_rows=' + raster.length);
  } catch (e) {
    $('status').textContent = '后端连接失败: ' + e + '（确认 pipeline --serve 已启动）';
  } finally {
    polling = false;
  }
}

function loop(t) {
  const dt = Math.min(0.05, (t - (lastT || 0)) / 1000 || 0.016); lastT = t || 0;
  updateVisuals(dt);
  fly.rotation.y = -0.35;   // 略侧身便于观察
  renderer.render(scene, camera);
}

function updateVisuals(dt) {
  // 翅膀: 幅度∝(flap_amp-0.5), 静息(0.5)几乎不扇, 只有 FLAP 才大幅扇动
  const wingDrive = Math.max(0, motor.flap_amp - 0.5) * 2;   // 0.5→0, 1.0→1
  wingPhase += dt * (6 + wingDrive * 40);
  const ang = Math.sin(wingPhase) * (0.05 + 1.1 * wingDrive);
  wingL.group.rotation.x = ang;
  wingR.group.rotation.x = -ang;
  // 六足: 幅度∝(leg_gain-0.5), 静息腿几乎不动, 只有被驱动的腿才迈步
  const g = motor.leg_gains || [];
  legNodes.forEach((leg, i) => {
    const gain = g[i] != null ? g[i] : 0.5;
    const legDrive = Math.max(0, gain - 0.5) * 2;            // 0.5→0, 1.0→1
    const tripod = (i === 0 || i === 2 || i === 4) ? 0 : Math.PI; // 对角三足
    const swing = Math.sin(wingPhase * 0.4 + tripod) * legDrive * 0.5;
    leg.hinge.rotation.z = leg.rest + swing;
  });
  // 神经元点亮度由放电率决定, 在 requestDecision 里已设好, 这里不再逐帧改动
}
