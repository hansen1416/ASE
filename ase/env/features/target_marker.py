# env/features/target_marker.py
from __future__ import annotations
import numpy as np
import torch

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import to_torch

from env.features.base import Feature
from env.features.task_types import HumanoidTask


class TargetMarkerFeature(Feature):
    """
    Red markers for visualizing reference key-body positions.

    Assumption (same as your current code):
      - actor 0 in each env is the humanoid
      - actors 1..K are the K markers (created immediately after humanoid)
    """

    def __init__(
        self,
        enabled: bool = True,
        asset_relpath: str = "urdf/traj_marker.urdf",
        color=(1.0, 0.0, 0.0),
        show_only_with_viewer: bool = True,
    ):
        self.enabled = enabled
        self._asset_relpath = asset_relpath
        self._color = color
        self._show_only_with_viewer = show_only_with_viewer

        # filled by hooks
        self._num_markers = 0
        self._marker_asset = None
        self._marker_handles_np = None  # [num_envs, K] int32

        self._marker_states = None      # view into task._root_states: [num_envs, K, 13]
        self._marker_pos = None         # view: [num_envs, K, 3]
        self._marker_actor_ids = None   # [num_envs*K] int32 (global actor ids)

        # PHC-side buffers (safe even if not used)
        self._vis_motion_ids = None     # [num_envs]
        self._vis_motion_times = None   # [num_envs]


    # ---- internals ----
    def _ensure_vis_buffers(self, task:HumanoidTask, env_ids = None) -> None:
        if self._vis_motion_ids is None or self._vis_motion_times is None:
            self._vis_motion_ids = torch.zeros(task.num_envs, device=task.device, dtype=torch.long)
            self._vis_motion_times = torch.zeros(task.num_envs, device=task.device, dtype=torch.float32)

        motion_ids = getattr(self, '_reset_ref_motion_ids', None)
        motion_times = getattr(self, '_reset_ref_motion_times', None)

        if motion_ids is not None and motion_times is not None and env_ids is not None:
            self._vis_motion_ids[env_ids] = self._reset_ref_motion_ids
            self._vis_motion_times[env_ids] = self._reset_ref_motion_times

    # ---- hooks ----
    def on_create_envs(self, task:HumanoidTask, num_envs: int) -> None:
        if not self.enabled:
            return

        self._num_markers = len(task.cfg["env"]["keyBodies"])
        self._marker_handles_np = np.zeros((num_envs, self._num_markers), dtype=np.int32)

        asset_root = task.cfg["env"]["asset"]["assetRoot"]
        opts = gymapi.AssetOptions()
        opts.fix_base_link = True
        opts.disable_gravity = True
        opts.angular_damping = 0.0
        opts.linear_damping = 0.0
        self._marker_asset = task.gym.load_asset(task.sim, asset_root, self._asset_relpath, opts)

    def on_humanoid_actor_created(self, task:HumanoidTask, env_id: int, env_ptr) -> None:
        if not self.enabled:
            return

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 1000.0)  # hidden by default

        for k in range(self._num_markers):
            h = task.gym.create_actor(
                env_ptr,
                self._marker_asset,
                pose,
                f"target_marker_{k}",
                env_id,
                0,
                0,
            )
            task.gym.set_rigid_body_color(
                env_ptr, h, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*self._color)
            )
            self._marker_handles_np[env_id, k] = int(h)
        

    def on_post_init_tensors(self, task:HumanoidTask) -> None:
        if not self.enabled:
            return

        self._ensure_vis_buffers(task)

        num_actors = task._root_states.shape[0] // task.num_envs
        root_view = task._root_states.view(task.num_envs, num_actors, task._root_states.shape[-1])

        # markers are actors 1..K
        self._marker_states = root_view[:, 1 : 1 + self._num_markers, :]
        self._marker_pos = self._marker_states[..., 0:3]

        # init hidden + identity rot
        self._marker_pos[:] = 1000.0
        self._marker_states[..., 3:7] = 0.0
        self._marker_states[..., 6] = 1.0

        marker_local_ids = to_torch(self._marker_handles_np, device=task.device, dtype=torch.int32)
        self._marker_actor_ids = (task._humanoid_actor_ids.unsqueeze(-1) + marker_local_ids).reshape(-1)

        # push once
        task.gym.set_actor_root_state_tensor_indexed(
            task.sim,
            gymtorch.unwrap_tensor(task._root_states),
            gymtorch.unwrap_tensor(self._marker_actor_ids),
            len(self._marker_actor_ids),
        )
        


    def on_reset_envs(self, task:HumanoidTask, env_ids) -> None:
        """
        One supporting fix: target_marker.py must not zero-out the sampled motion start time

        Your marker feature currently does:

        _ensure_vis_buffers() copies sampled motion ids/times (good)

        on_reset_envs() overwrites _vis_motion_times[env_ids] = 0.0 (bad if you sample nonzero start times). 

        target_marker

        Change in TargetMarkerFeature.on_reset_envs():

        # remove this line:
        # self._vis_motion_times[env_ids] = 0.0
        # keep the times set by _ensure_vis_buffers()

        This makes markers stay aligned with the same motion_start_times you use for task obs.
        """


        if not self.enabled:
            return
        self._ensure_vis_buffers(task)
        # reset visual time for the envs that reset

        # todo AMP OBS, we need to consider random start
        self._vis_motion_times[env_ids] = 0.0

        if self._show_only_with_viewer and task.viewer is None:
            return
        # show immediately on reset frame (optional)
        self._update_markers(task, env_ids=env_ids, use_next_step=False)
        

    def on_post_physics_step(self, task:HumanoidTask) -> None:
        if not self.enabled:
            return
        if self._show_only_with_viewer and task.viewer is None:
            return

        self._vis_motion_times += task.dt
        self._update_markers(task, env_ids=None, use_next_step=True)
        

    def _update_markers(self, task:HumanoidTask, env_ids=None, use_next_step: bool = True) -> None:
        # needs MotionLib (HumanoidPHC provides task._motion_lib)
        motion_lib = getattr(task, "_motion_lib", None)
        if motion_lib is None or self._marker_pos is None:
            return

        if env_ids is None or len(env_ids) == 0:
            env_ids = torch.arange(task.num_envs, device=task.device, dtype=torch.long)

        t = self._vis_motion_times[env_ids]
        if use_next_step:
            t = t + task.dt

        motion_ids = self._vis_motion_ids[env_ids]
        *_, key_pos = motion_lib.get_motion_state(motion_ids, t)

        self._marker_pos[env_ids] = key_pos  # [N, K, 3]

        marker_ids = self._marker_actor_ids.view(task.num_envs, self._num_markers)[env_ids].reshape(-1)
        task.gym.set_actor_root_state_tensor_indexed(
            task.sim,
            gymtorch.unwrap_tensor(task._root_states),
            gymtorch.unwrap_tensor(marker_ids),
            len(marker_ids),
        )
