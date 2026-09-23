"""把 web/model 下的 .bin 组装成一只果蝇, 单视角实心渲染确认几何正确。"""
import json, struct, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT = "/Users/shenjie/fly-brain-agent/web/model/"
mf = json.load(open(OUT + "fly_manifest.json"))

def load(fn):
    b = open(OUT + fn, "rb").read()
    n = struct.unpack("<i", b[:4])[0]
    a = np.frombuffer(b[4:], dtype="<f4").reshape(n, 6)
    return a[:, :3].reshape(-1, 3, 3)  # (tris,3,3) positions

parts = [
    ("body", mf["body"]["file"], (0.62, 0.42, 0.14), 1.0),
    ("eye",  mf["eye"]["file"],  (0.70, 0.22, 0.12), 1.0),
    ("wl",   mf["wing_l"]["file"],(0.80, 0.80, 0.92), 0.4),
    ("wr",   mf["wing_r"]["file"],(0.80, 0.80, 0.92), 0.4),
]

fig = plt.figure(figsize=(14, 6))
for pi, (az, el) in enumerate([(-60, 20), (-90, 90), (0, 0)]):
    ax = fig.add_subplot(1, 3, pi + 1, projection="3d")
    allpts = []
    for name, fn, col, alpha in parts:
        tri = load(fn)
        allpts.append(tri.reshape(-1, 3))
        pc = Poly3DCollection(tri, facecolor=col, edgecolor="none", alpha=alpha)
        ax.add_collection3d(pc)
    P = np.concatenate(allpts)
    c = P.mean(0); r = (P.max(0) - P.min(0)).max() / 2
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=el, azim=az)
    ax.set_title(f"view {pi+1}"); ax.set_axis_off()
fig.tight_layout()
fig.savefig("/Users/shenjie/fly-brain-agent/render_check.png", dpi=90)
print("saved render_check.png")
