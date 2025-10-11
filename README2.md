motion_aa
Per-joint axis–angle rotations for the whole pose at the queried time(s). Useful for kinematic reconstruction, rendering, or losses that operate in joint-angle space. ASE doesn’t need it at runtime (it uses dof_pos directly).

rg_pos (“rigid group”/global positions)
Global 3D positions of all articulated bodies/nodes for the motion frames (shape: [..., n_bodies, 3]). PHC keeps the full body trajectory for visualization, contact checks, and arbitrary keypoint sampling. ASE only asks for a subset (key_pos) so it doesn’t return the full tensor.

rb_rot (rigid-body global rotations)
Global orientations (quaternions) for all bodies (shape: [..., n_bodies, 4]). Same rationale as rg_pos.

body_vel
Linear velocities for all bodies (not just the root). Helpful for dynamics-aware objectives or diagnostics. ASE only needs the root linear velocity.

body_ang_vel
Angular velocities for all bodies. Again, PHC keeps the full field; ASE only needs the root angular velocity.

motion_bodies
The ordered list/indices of bodies that the motion file defines. PHC exposes it so downstream code can map names↔indices or pick subsets on the fly. In ASE that mapping is baked into the task (key_body_ids, dof_body_ids) so it isn’t re-emitted.

motion_limb_weights
Per-body (or per-limb) weights used in PHC for limb-aware losses/blending (e.g., up-weight feet for contact, down-weight fingers), curriculum, or quality metrics. ASE’s MotionLib API doesn’t include this; similar weighting (if any) is usually internal to the reward/discriminator.