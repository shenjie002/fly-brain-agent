// ============================================================
//  FlyBrain Flappy — 3D NeuroMechFly + 双轨(Laya→LIF→FlyGym→Bird) 闭环
//  链路(每帧)：
//   [1] laya.decide(鸟态) -> {action, confidence}     Laya 本地快脑
//   [2] net.run(鸟态, laya) -> {flap}                 神经 LIF 反馈
//   [3] flap 驱动：FlyGym 果蝇扇翅/flap, + 小鸟上升
//   [4] 置信 < 阈值 -> net 长程 consult 校准(慢轨) 再交回
// ============================================================
import * as THREE from 'three';
import { LayaBrain, NeuroNet } from './brain.js';

const $ = (id) => document.getElementById(id);
const ASSET = './model/';
const clamp = (x, a, b) => x < a ? a : (x > b ? b : x);
const rand = (a, b) => a + Math.random() * (b - a);

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

// ---- 物理参数 (世界 +y 上) ----
const GRAV = 11.0, FLAP_V = 6.6, PS = 3.2, PIPE_W = 1.4, GAP = 4.6,
      FLY_X = -6.0, SPAWN = 24, SPACING = 8, Y_MIN = -4.6, Y_MAX = 6.6,
      SLOWTH = 0.34;

let renderer, scene, camera, fly, wingL, wingR, pipes = [],
    phase = 'ready', score = 0, best = 0, elapsed = 0, wingPhase = 0, curFlap = 0,
    slow = false, slowT = 0, lastConf = 0.5, lastAction = '—',
    birdY = 0, birdVy = 0, gapCentre = 0, lastT = 0;
const laya = new LayaBrain(), net = new NeuroNet();

try { best = +(localStorage.getItem('flybrain_best') || 0); } catch (_) {}

init().catch((e) => { console.error(e); const o = document.querySelector('#ov-msg'); if (o) o.textContent = '加载失败: ' + e; });

// ---------------------------------------------------------------------------
async function init() {
  const mf = await (await fetch(ASSET + 'fly_manifest.json')).json();

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  $('stage').appendChild(renderer.domElement);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0f3550);
  scene.fog = new THREE.Fog(0x0f3550, 32, 66);

  camera = new THREE.PerspectiveCamera(52, innerWidth / innerHeight, 0.1, 120);
  camera.up.set(0, 1, 0);
  scene.add(new THREE.AmbientLight(0xffffff, 0.85));
  const key = new THREE.DirectionalLight(0xffffff, 1.2); key.position.set(-5, 9, 9); scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, 0.5); fill.position.set(6, -2, -7); scene.add(fill);

  // 组装果蝇
  fly = new THREE.Group(); fly.scale.setScalar(1.6);
  fly.add(new THREE.Mesh(await loadGeo(mf.body), matf(mf.body_rgba)));
  fly.add(new THREE.Mesh(await loadGeo(mf.eye), matf(mf.eye_rgba)));
  wingL = wingNode(await loadGeo(mf.wing_l), matf(mf.wing_rgba), mf.wing_l.root);
  wingR = wingNode(await loadGeo(mf.wing_r), matf(mf.wing_rgba), mf.wing_r.root);
  fly.add(wingL.group, wingR.group);
  fly.position.set(FLY_X, 0, 0);
  scene.add(fly);

  buildPipes();
  buildGround();

  $('ov-btn').classList.remove('hidden');
  $('ov-btn').onclick = run;
  $('ov-title').textContent = 'FlyBrain Flappy';
  $('ov-msg').textContent = 'Laya 快脑 → LIF 神经 → 果蝇 · 小鸟同步';

  addEventListener('keydown', (e) => {
    if (e.code === 'Space') { e.preventDefault(); if (phase === 'ready' || phase === 'over') run(); else if (phase === 'live') tap(); }
  });
  addEventListener('pointerdown', () => { if (phase === 'ready' || phase === 'over') run(); else if (phase === 'live') tap(); });
  addEventListener('resize', () => { renderer.setSize(innerWidth, innerHeight); camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); });
  document.querySelectorAll('#cam button').forEach((x) => {
    x.onclick = () => {
      document.querySelectorAll('#cam button').forEach((y) => y.classList.remove('active'));
      x.classList.add('active');
      if (x.dataset.cam === 'top') { camera.position.set(0, 18, 2); camera.up.set(0, 0, 1); camera.lookAt(0, 0, 0); }
      else if (x.dataset.cam === 'side') { camera.position.set(-8, 2, 16); camera.up.set(0, 1, 0); camera.lookAt(0, 0, 0); }
      else { camera.position.set(-9, 5, 18); camera.up.set(0, 1, 0); camera.lookAt(0, 0, 0); }
    };
  });

  camera.position.set(-9, 5, 18); camera.lookAt(0, 0, 0);
  renderer.setAnimationLoop(loop);
}

function wingNode(geo, mat, base) {
  const group = new THREE.Group();
  const m = new THREE.Mesh(geo.clone(), mat);
  m.geometry.translate(-base[0], -base[1], -base[2]);
  group.add(m);
  group.position.set(base[0], base[1], base[2]);
  return { group };
}

function buildPipes() {
  const gc = new THREE.MeshStandardMaterial({ color: 0x2f8f3f, side: THREE.DoubleSide });
  const cc = new THREE.MeshStandardMaterial({ color: 0x3f9f4f, side: THREE.DoubleSide });
  for (let i = 0; i < 6; i++) {
    const g = new THREE.Group();
    const top = new THREE.Mesh(new THREE.BoxGeometry(PIPE_W, 16, 1.5), gc);
    const bot = new THREE.Mesh(new THREE.BoxGeometry(PIPE_W, 16, 1.5), gc);
    const lt = new THREE.Mesh(new THREE.BoxGeometry(PIPE_W + 0.4, 1.6, 1.7), cc);
    const lb = new THREE.Mesh(new THREE.BoxGeometry(PIPE_W + 0.4, 1.6, 1.7), cc);
    top.add(lt); bot.add(lb);
    g.add(top, bot);
    scene.add(g);
    pipes.push({ group: g, top, bot, x: 0, gy: 0, passed: false });
  }
  layout(true);
}

function layout(reset) {
  for (let i = 0; i < pipes.length; i++) {
    if (reset) { pipes[i].x = SPAWN + i * SPACING; pipes[i].gy = rand(-3.5, 4); pipes[i].passed = false; }
    pipes[i].group.position.set(pipes[i].x, pipes[i].gy, 0);
    pipes[i].top.position.y = GAP / 2 + 1.6;
    pipes[i].bot.position.y = -GAP / 2 - 1.6;
  }
}

function buildGround() {
  const g = new THREE.Mesh(new THREE.PlaneGeometry(600, 400), new THREE.MeshBasicMaterial({ color: 0x0a2410, side: THREE.DoubleSide }));
  g.rotation.x = -Math.PI / 2; g.position.y = Y_MIN - 1.5; scene.add(g);
}

// ---- 状态机 ----
function run() {
  $('overlay').classList.add('hidden'); phase = 'live'; score = 0;
  birdY = 0; birdVy = 0; curFlap = 0; slow = false; slowT = 0;
  layout(true);
}
function tap() { birdVy = FLAP_V; }
function nextGap() { let b = null; for (const p of pipes) if (p.x > FLY_X && (!b || p.x < b.x)) b = p; return b || pipes[0]; }

function dead() {
  phase = 'over';
  $('ov-title').textContent = '撞线！Over';
  $('ov-msg').textContent = '得分 ' + score + ' · 最佳 ' + best;
  $('ov-btn').textContent = '再试一次';
  $('ov-btn').onclick = run;
  $('overlay').classList.remove('hidden');
  try { localStorage.setItem('flybrain_best', best); } catch (_) {}
}

function loop(t) {
  const dt = Math.min(0.04, (t - (lastT || 0)) / 1000 || 0.016); lastT = t || 0;
  if (phase === 'live') step(dt);
  updateVisuals(dt);
  updateHUD();
  renderer.render(scene, camera);
}

function step(dt) {
  elapsed += dt;
  birdVy -= GRAV * dt; birdY += birdVy * dt;
  if (birdY < Y_MIN) { birdY = Y_MIN; birdVy = 0; }
  if (birdY > Y_MAX) { birdY = Y_MAX; birdVy = 0; }

  const gap = nextGap(); gapCentre = gap.gy;
  for (const p of pipes) {
    p.x -= PS * dt; p.group.position.x = p.x;
    if (!p.passed && p.x < FLY_X) { p.passed = true; score++; best = Math.max(best, score); try { localStorage.setItem('flybrain_best', best); } catch (_) {} }
    if (p.x < -22) { p.x = SPAWN + SPACING; p.gy = rand(-3.5, 4); p.passed = false; p.group.position.set(p.x, p.gy, 0); p.top.position.y = GAP / 2 + 1.6; p.bot.position.y = -GAP / 2 - 1.6; }
  }

  // [1] Laya 本地快脑
  const tNext = Math.max(0, (gap.x - FLY_X) / PS);
  const st = { birdY, vy: birdVy, gapCentreY: gap.gy, gapHalf: GAP / 2, tNext };
  const L = laya.decide(st);
  lastAction = L.action; lastConf = L.confidence;

  // [2] 神经 LIF 反馈 (内部跑 ms 毫秒; 慢轨用更长)
  const nf = net.run(st, L, slow ? 24 : 8);
  curFlap = nf.flap;

  // [4] 慢轨: 低置信 -> 长程 consult
  if (!slow && lastConf < SLOWTH) { slow = true; slowT = 0.34; }
  if (slow) {
    slowT -= dt;
    const nfL = net.run(st, L, 30);
    curFlap = Math.max(curFlap, nfL.flap);
    if (slowT <= 0) slow = false;
  }

  // [3] 神经驱动 -> 小鸟
  if (curFlap > 0.5) birdVy = Math.max(birdVy, FLAP_V * 0.6);
  fly.position.y = birdY;

  if (collide()) dead();
}

function collide() {
  for (const p of pipes) {
    if (Math.abs(p.x - FLY_X) < PIPE_W / 2 + 0.35) {
      if (birdY > p.gy + GAP / 2 - 0.5 || birdY < p.gy - GAP / 2 + 0.5) return true;
    }
  }
  return false;
}

function updateVisuals(dt) {
  const amp = slow ? 1.7 : 1.0;
  wingPhase += dt * (curFlap > 0.5 ? 26 : 9);
  const ang = Math.sin(wingPhase) * (0.5 + 0.7 * curFlap) * amp;
  wingL.group.rotation.x = ang;
  wingR.group.rotation.x = -ang;
  fly.rotation.z = clamp(-birdVy * 0.05, -0.5, 0.5);
  if (slow) fly.rotation.z += Math.sin(wingPhase * 10) > 0 ? 0.05 : -0.05;
}

function updateHUD() {
  $('score').textContent = score;
  $('best').textContent = 'Best: ' + (best || '—');
  $('act').textContent = lastAction;
  $('confnum').textContent = lastConf.toFixed(2);
  const fill = $('confbar').firstElementChild; if (fill) fill.style.width = (lastConf * 100) + '%';
  const dot = $('brain-dot'); if (dot) dot.classList.toggle('slow', slow);
}