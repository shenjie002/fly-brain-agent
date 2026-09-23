// ============================================================
//  Brain.cjs — 本地"快慢双轨"决策链路 (Laya → LIF 神经 → 动作)
//  用户链路:
//   [1] Laya 快脑(decide): 读小鸟状态 -> choice(FLAP/HOLD)+confidence
//   [2] Laya 把决策 反馈给 神经(LIF 网络, respond)
//   [3] 神经 LIF 整合 -> 输出 flap 增益 (驱动 FlyGym 果蝇扇翅 + 小鸟上升)
//   [4] 慢轨: confidence 低 -> consultLong 长程 consult(更多毫秒) 校准安全
// ============================================================

const NOTE=[];
export function note(m){NOTE.push(String(m)); if(NOTE.length>26)NOTE.shift();}
export function notes(){return NOTE.slice();}
const cl=(x,a,b)=>x<a?a:(x>b?b:x);
const sig=(raw,k=5.0)=>1/(1+Math.exp(-Math.abs(raw)*k));

// ------------------------------------------------------------
// Laya 快脑: 读小鸟状态 → 决策
//   st = { birdY, vy, gapCentreY, gapHalf, tNext(sec) }
//   +y 向上.  gapCentreY > birdY -> 缝隙在上面 -> 需升 -> FLAP
// ------------------------------------------------------------
export class LayaBrain{
  constructor(){ this.kGap=2.6; this.kVy=1.1; this.kUrG=0.9; }
  decide(st){
    const dt=cl(st.tNext,0,1.1);
    const pred=st.birdY+st.vy*dt*0.55;          // 在缝隙处的预测高度
    const need=st.gapCentreY-pred;                     // + 需升
    let raw=this.kGap*(need>0?need:0.22*need)
           -this.kVy*Math.max(0,st.vy)
           +this.kUrG*(dt<0.3?(0.3-dt)/0.3:0)*0.5;
    const action=raw>0.05?'FLAP':'HOLD';
    return { action, raw, confidence: this.conf(raw,st) };
  }
  conf(raw,st){
    const off=Math.abs(st.birdY-st.gapCentreY);
    const edge=cl(off/(st.gapHalf||1),0,1.4);
    return sig(raw)*(1-0.35*Math.max(0,edge-1));
  }
}

// ------------------------------------------------------------
// 神经: LIF 网络 (simulator.py 逻辑移植), N=6
//   通道: 0 需升 / 1 需降 / 2 紧迫 / 3 偏置 / 4 决策集成 / 5 输出(filter)
//   consult(LayaOut) 跑 ms 毫秒 -> return { flap:0..1 }
// ------------------------------------------------------------
export class NeuroNet{
  constructor(){
    this.N=6; this.R=2.4; this.dt=1; this.tau=10;
    this.decay=Math.exp(-this.dt/this.tau);
    this.th=-55; this.vr=-68;
    this.W=[
      [0,0,0,0,1.4,0],
      [0,0,0,0,-0.8,0],
      [0,0,0,0,0.7,0],
      [0,0,0,0,1.8,0],
      [0,0,0,0,0,1.1],
      [0,0,0,0,0,0],
    ];
    this.reset();
  }
  reset(){
    this.v=new Array(this.N).fill(-65);
    this.last=new Array(this.N).fill(0);
    this.flap=0; this.spikes4=0; this.spikes5=0;
  }
  _step(ext){
    const sp=this.last.slice();
    for(let i=0;i<this.N;i++){
      let syn=0; for(let j=0;j<this.N;j++) syn+=this.W[j][i]*sp[j];
      this.v[i]=this.v[i]*this.decay+this.R*(ext[i]+syn);
    }
    const fired=[];
    for(let i=0;i<this.N;i++) if(this.v[i]>=this.th){fired.push(i);this.v[i]=this.vr;}
    this.last=sp.map((_,i)=>fired.includes(i)?1:0);
    if(fired.includes(4))this.spikes4++;
    if(fired.includes(5))this.spikes5++;
  }
  // st=世界态, lap= Laya 输出, ms= 神经整合用毫秒数
  run(st, laya, ms=8){
    this.reset();
    const need=cl(st.gapCentreY-st.birdY,-1,1);
    const urg=cl(1-st.tNext,0,1);
    const flapCmd=laya&&laya.action==='FLAP';
    for(let i=0;i<ms;i++){
      const ext=[
        need>0 ? need*2.0+(flapCmd?1.4:0) : 0,
        need<0 ? -need*2.0 : 0,
        urg*1.2, 0.5, 0, 0,
      ];
      this._step(ext);
    }
    // 神经输出: 继承通道5净脉冲 -> 0..1 flap 增益
    this.flap= this.spikes5> 2 ? 1.0 : (this.spikes5>0?0.7:0);
    return { flap:this.flap, spikes4:this.spikes4, spikes5:this.spikes5 };
  }
  // 慢轨长程 consult: 更多毫秒
  runLong(st, laya){ return this.run(st, laya, 24); }
}