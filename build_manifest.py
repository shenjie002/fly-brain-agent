"""Bake the wasm-game fly into web-ready binary geometry for a Three.js Flappy fly.

Silhouette (thorax, head, eyes, abdomen) is merged into ONE non-indexed
BufferGeometry (world-space verts, recomputed normals). The two wings are
written separately so they can be flapped around their root. Right-side parts
are taken from MuJoCo's geometry which already carries the mirror scale, so the
results are pixel-faithful to the real NeuroMechFly pose.

Outputs under web/model/:
  fly.body.bin  : [int32 n][pos f32*3 | normal f32*3]  (n/3 triangles)
  wing_l.bin    : same for left  wing mesh
  wing_r.bin    : same for right wing mesh
  fly_manifest.json : counts + rgba + wing root anchors
"""
import json, struct, numpy as np, mujoco

XML = "/Users/shenjie/Downloads/flygym-2.1.0/wasm/game/assets/model/fly.xml"
OUT = "/Users/shenjie/fly-brain-agent/web/model/"

BODY = [0.62, 0.42, 0.14, 1.0]
EYE  = [0.70, 0.22, 0.12, 1.0]
WING = [0.85, 0.85, 0.95, 0.35]

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)


def mname(m, dataid):
    return m.names[m.name_meshadr[dataid]:].split(b"\x00")[0].decode()


def gname(m, i):
    return m.names[m.geom_nameadr[i]:].split(b"\x00")[0].decode()


def verts_of(m, dataid):
    """Return the mesh as a non-indexed triangle soup (faces expanded).

    mesh_vert is the DEDUPLICATED vertex table; the triangle connectivity lives
    in mesh_face (indices into that table). Reading mesh_vert alone and treating
    every 3 consecutive verts as a triangle scrambles the geometry into shards,
    so we must expand each face's 3 indices into real vertices.
    """
    va = m.mesh_vertadr[dataid]
    vn = m.mesh_vertnum[dataid]
    verts = np.asarray(m.mesh_vert[va * 3: va * 3 + vn * 3]).reshape(-1, 3)
    fa = m.mesh_faceadr[dataid]
    fn = m.mesh_facenum[dataid]
    faces = np.asarray(m.mesh_face[fa * 3: fa * 3 + fn * 3]).reshape(-1, 3)
    return verts[faces].reshape(-1, 3)   # (fn*3, 3) triangle soup


def normals(v3):
    v = v3.reshape(-1, 9)
    a, b, c = v[:, 0:3], v[:, 3:6], v[:, 6:9]
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    per_tri = (n / ln)
    return np.repeat(per_tri, 3, axis=0).astype(np.float32)


def write_bin(path, verts):
    verts = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
    n = verts.shape[0]
    blob = np.concatenate([verts, normals(verts)], axis=1)
    with open(path, "wb") as f:
        f.write(struct.pack("<i", n))
        f.write(blob.tobytes())
    return n


def short_class(short):
    if short in ("l_eye", "r_eye"):
        return "eye"
    if short in ("l_wing", "r_wing"):
        return "wing"
    return "body"


body = []
eyes = []
wings = {}
for g in range(m.ngeom):
    if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
        continue
    data = m.geom_dataid[g]
    if data < 0:
        continue
    mesh = mname(m, data)
    short = mesh.split("/")[-1]
    keep = short in ("c_thorax", "c_head", "l_eye", "r_eye",
                     "c_abdomen12", "c_abdomen3", "c_abdomen4",
                     "c_abdomen5", "c_abdomen6") or short in ("l_wing", "r_wing")
    if not keep:
        continue
    v = verts_of(m, data)
    R = np.asarray(d.geom_xmat[g]).reshape(3, 3)
    t = np.asarray(d.geom_xpos[g])
    world = v @ R.T + t
    cls = short_class(short)
    if cls == "wing":
        wings[short] = world
    elif cls == "eye":
        eyes.append(world)
    else:
        body.append(world)


body = np.concatenate(body, axis=0) if body else np.zeros((1, 3))
eyes = np.concatenate(eyes, axis=0) if eyes else np.zeros((1, 3))
nb = write_bin(OUT + "fly.body.bin", body)
ne = write_bin(OUT + "eye.bin", eyes)
wl = wings.get("l_wing", np.zeros((3, 3)))
wr = wings.get("r_wing", np.zeros((3, 3)))
nwl = write_bin(OUT + "wing_l.bin", wl)
nwr = write_bin(OUT + "wing_r.bin", wr)


def wing_root(v, side):
    if v.shape[0] < 3:
        return [0.0, 0.0, 0.0]
    span = np.max(v, axis=0) - np.min(v, axis=0)
    axis = int(np.argmax(span))
    col = v[:, axis]
    idx = np.argmin(np.abs(col)) if side == "l" else np.argmax(np.abs(col))
    return [float(x) for x in v[idx]]


manifest = {
    "body":    {"file": "fly.body.bin", "count": body.shape[0]},
    "eye":     {"file": "eye.bin", "count": eyes.shape[0]},
    "wing_l":  {"file": "wing_l.bin", "count": nwl,
                "root": wing_root(wl, "l")},
    "wing_r":  {"file": "wing_r.bin", "count": nwr,
                "root": wing_root(wr, "r")},
    "body_rgba": BODY, "eye_rgba": EYE, "wing_rgba": WING,
}
with open(OUT + "fly_manifest.json", "w") as f:
    json.dump(manifest, f, indent=1)
print("fly.body.bin", body.shape[0] // 3, "tris, wings", nwl, nwr, "-> manifest done")