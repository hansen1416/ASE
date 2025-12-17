import mujoco

xml_path = "ase/data/assets/mjcf/smpl/aaab922b_smpl.xml"   # your two faulty ones
# xml_path = "ase/data/assets/mjcf/smpl/6803e1fa_smpl.xml"   # your two faulty ones

# xml_path = "ase/data/assets/mjcf/smpl/75e01b05_smpl.xml"

m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)

# free joint: [x, y, z, qw, qx, qy, qz]
d.qpos[0:3] = [0.0, 0.0, 1.0]
d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

mujoco.mj_forward(m, d)

# Report deep penetrations at initialization
pairs = []
for i in range(d.ncon):
    c = d.contact[i]
    if c.dist < -1e-3:  # penetration
        g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        pairs.append((c.dist, g1, g2))
pairs.sort()

for dist, g1, g2 in pairs[:30]:
    print(f"{dist: .6f}  {g1}  <->  {g2}")
