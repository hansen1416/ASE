import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from utils import torch_utils
from env.tasks.base_task import BaseTask
from env.features.target_marker import TargetMarkerFeature

SMPL_MUJOCO_NAMES = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 
                     'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
SMPLH_MUJOCO_NAMES = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 
                      'L_Wrist', 'L_Index1', 'L_Index2', 'L_Index3', 'L_Middle1', 'L_Middle2', 'L_Middle3', 'L_Pinky1', 'L_Pinky2', 'L_Pinky3', 'L_Ring1', 'L_Ring2', 'L_Ring3', 'L_Thumb1', 'L_Thumb2', 'L_Thumb3', 
                      'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Index1', 'R_Index2', 'R_Index3', 'R_Middle1', 'R_Middle2', 'R_Middle3', 'R_Pinky1', 'R_Pinky2', 'R_Pinky3', 'R_Ring1', 'R_Ring2', 'R_Ring3', 'R_Thumb1', 'R_Thumb2', 'R_Thumb3']

def mat33_to_np(m):
    # m: gymapi.Mat33
    return np.array([
        [m.x.x, m.x.y, m.x.z],
        [m.y.x, m.y.y, m.y.z],
        [m.z.x, m.z.y, m.z.z],
    ], dtype=np.float32)


class Humanoid(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine

        self._pd_control = self.cfg["env"]["pdControl"]
        self.power_scale = self.cfg["env"]["powerScale"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self._local_root_obs = self.cfg["env"]["localRootObs"]
        self._root_height_obs = self.cfg["env"].get("rootHeightObs", True)
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]
        
        key_bodies = self.cfg["env"]["keyBodies"]
        self._setup_character_props(key_bodies)

        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless
        
        # marker logic -------------
        # self._enable_target_markers = True
        # marker logic -------------

        # fetures plugin -------------
        self._features = [TargetMarkerFeature(enabled=True)]
        # fetures plugin -------------
         
        super().__init__(cfg=self.cfg)
        
        self.dt = self.control_freq_inv * sim_params.dt
        
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        # multi humanoid template change ===============
        self.force_sensor_joints = cfg["env"].get("force_sensor_joints", ["L_Ankle", "R_Ankle"]) # force tensor joints
        sensors_per_env = len(self.force_sensor_joints)
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        num_actors = self.get_num_actors_per_env()
        
        # is a view (no copy) that reshapes self._root_states to [num_envs, num_actors_per_env, 13] 
        # and then selects the 0-th actor per env (the humanoid):
        # Because it is a view, in-place writes to self._humanoid_root_states[:, 0:3/3:7/…] 
        # directly modify the corresponding slice of self._root_states.
        self._humanoid_root_states = self._root_states.view(self.num_envs, num_actors, actor_root_state.shape[-1])[..., 0, :]
        self._initial_humanoid_root_states = self._humanoid_root_states.clone()
        self._initial_humanoid_root_states[:, 7:13] = 0

        self._humanoid_actor_ids = num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.int32)
        
        # fetures plugin -------------
        for f in self._features: f.on_post_init_tensors(self)
        # fetures plugin -------------

        # marker logic -------------
        # if self._enable_target_markers:
        #     root_view = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])

        #     # Assumption (kept simple): humanoid is actor 0, markers are actors 1..K in each env
        #     self._target_marker_states = root_view[:, 1:1 + self._num_target_markers, :]
        #     self._target_marker_pos = self._target_marker_states[..., 0:3]

        #     # hide initially
        #     self._target_marker_pos[:] = 1000.0
        #     self._target_marker_states[..., 3:7] = 0.0  # w component (x,y,z,w)
        #     self._target_marker_states[..., 6] = 1.0

        #     marker_local_ids = to_torch(self._target_marker_handles, device=self.device, dtype=torch.int32)
        #     self._target_marker_actor_ids = (self._humanoid_actor_ids.unsqueeze(-1) + marker_local_ids).reshape(-1)


        #     self.gym.set_actor_root_state_tensor_indexed(
        #         self.sim,
        #         gymtorch.unwrap_tensor(self._root_states),
        #         gymtorch.unwrap_tensor(self._target_marker_actor_ids),
        #         len(self._target_marker_actor_ids)
        #     )
        # marker logic -------------

        # create some wrapper tensors for different slices
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        dofs_per_env = self._dof_state.shape[0] // self.num_envs
        self._dof_pos = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 0]
        self._dof_vel = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 1]
        
        self._initial_dof_pos = torch.zeros_like(self._dof_pos, device=self.device, dtype=torch.float)
        self._initial_dof_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)
        
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)

        self._rigid_body_pos = rigid_body_state_reshaped[..., :self.num_bodies, 0:3]
        self._rigid_body_rot = rigid_body_state_reshaped[..., :self.num_bodies, 3:7]
        self._rigid_body_vel = rigid_body_state_reshaped[..., :self.num_bodies, 7:10]
        self._rigid_body_ang_vel = rigid_body_state_reshaped[..., :self.num_bodies, 10:13]

        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., :self.num_bodies, :]
        
        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        
        self._build_termination_heights()
        
        contact_bodies = self.cfg["env"]["contactBodies"]
        # `self._key_body_ids` later used to compute `_compute_amp_observations`
        # also MotionLib is configured to output only those key bodies `self._key_body_ids`
        self._key_body_ids = self._build_key_body_ids_tensor(key_bodies)
        self._contact_body_ids = self._build_contact_body_ids_tensor(contact_bodies)
        
        if self.viewer != None:
            self._init_camera()

        return

    def get_obs_size(self):
        return self._num_obs

    def get_action_size(self):
        return self._num_actions

    def get_num_actors_per_env(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        return num_actors

    def create_sim(self):
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        return

    def reset(self, env_ids=None):
        if (env_ids is None):
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        self._reset_envs(env_ids)
        return

    def set_char_color(self, col, env_ids):
        for env_id in env_ids:
            env_ptr = self.envs[env_id]
            handle = self.humanoid_handles[env_id]

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                              gymapi.Vec3(col[0], col[1], col[2]))

        return

    def _reset_envs(self, env_ids):
        if (len(env_ids) > 0):
            self._reset_actors(env_ids)
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()
            self._compute_observations(env_ids)
        return

    def _reset_env_tensors(self, env_ids):
        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        dof_pos = self._dof_state[..., :, 0]
        dof_pos = dof_pos.contiguous()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                      gymtorch.unwrap_tensor(dof_pos),
                                                      gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        
        return

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
        return

    def _setup_character_props(self, key_bodies):
        # multi humanoid template change ===============
        self._body_names = SMPL_MUJOCO_NAMES
        self._dof_names = self._body_names[1:]

        # ankle joints are the lowest articulated joints
        # self.force_sensor_joints = ["L_Ankle", "R_Ankle"]
        self._dof_body_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        self._dof_offsets = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69]

        self._dof_obs_size = len(self._dof_names) * 6
        self._dof_size = len(self._dof_names) * 3

        self._num_actions = len(self._dof_names) * 3

        # some conditions for `self._num_obs`, burrowed from PHC
        self._root_height_obs = True
        self.self_obs_v = 1

        # height + num_bodies * 15 (pos + vel + rot + ang_vel) - root_pos
        self._num_obs = 1 + len(self._body_names) * (3 + 6 + 3 + 3) - 3

        self._num_obs += 10

        # load beta into observation ===============

        if not self._root_height_obs:
            self._num_obs -= 1
        
        if self.self_obs_v == 3:
            self._num_obs += 6 * len(self.force_sensor_joints)

        return

    def _build_termination_heights(self):
        head_term_height = 0.3
        shield_term_height = 0.32

        termination_height = self.cfg["env"]["terminationHeight"]
        self._termination_heights = np.array([termination_height] * self.num_bodies)

        head_id = self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], "head")
        self._termination_heights[head_id] = max(head_term_height, self._termination_heights[head_id])

        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        if (asset_file == "mjcf/amp_humanoid_sword_shield.xml"):
            left_arm_id = self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], "left_lower_arm")
            self._termination_heights[left_arm_id] = max(shield_term_height, self._termination_heights[left_arm_id])
        
        self._termination_heights = to_torch(self._termination_heights, device=self.device)
        return

    def _compute_safe_root_height(self, template_id: int) -> float:
        """
        ---- 1211 actions
        Heuristic: use the first SMPL beta (roughly correlated with height)
        to scale a base height, then add a small margin. This is just a
        robust guess; tune coefficients empirically.
        """
        betas = self._template_betas[template_id]          # [B]
        beta_height = float(betas[0].clamp(-3.0, 3.0))     # avoid extremes
        height_scale = 1.0 + 0.15 * beta_height            # ~±45% over ±3
        safe_h = self._base_char_height * height_scale + self._spawn_height_margin
        # avoid silly values
        safe_h = max(0.4, min(2.0, safe_h))
        return safe_h

    def _create_envs(self, num_envs, spacing, num_per_row):
        # fetures plugin -------------
        for f in self._features: f.on_create_envs(self, num_envs)
        # fetures plugin -------------

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        asset_file = self.cfg["env"]["asset"]["assetFileName"]

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        
        # marker logic -------------
        # if self._enable_target_markers:
            
        #     self._num_target_markers = len(self.cfg["env"]["keyBodies"])
        #     self._target_marker_handles = np.zeros((num_envs, self._num_target_markers), dtype=np.int32)
        #     self._load_target_marker_asset()
        # marker logic -------------

        # # ---- 1211 actions robust convex decomposition & inertia overrides ----
        # # Use VHACD (volumetric hierarchical approximate convex decomposition)
        # # so that arbitrary meshes are approximated by a set of convex shapes.
        # # This tends to produce more stable contact behavior than raw triangle meshes.
        # asset_options.vhacd_enabled = True

        # # Ignore the center-of-mass from the imported asset and recompute it
        # # from the (possibly VHACD-processed) collision geometry. This keeps
        # # COM consistent with the new convex shapes and improves balance.
        # asset_options.override_com = True

        # # Ignore the inertia tensor from the imported asset and recompute it
        # # from the processed collision geometry. This avoids pathological
        # # inertia values when the original mesh or scaling is irregular.
        # asset_options.override_inertia = True

        # # Merge rigid bodies that are connected by fixed joints into a single
        # # rigid body where possible. This reduces joint count and can remove
        # # tiny, jitter-prone segments, leading to more stable simulation.
        # asset_options.collapse_fixed_joints = True

        # # Automatically replace cylinders in the collision geometry with
        # # capsules, which generally have more robust contact behavior and
        # # fewer edge cases in PhysX than cylinders.
        # asset_options.replace_cylinder_with_capsule = True
        # # ---- 1211 actions ---------------------------------------------------------

        # multi humanoid template change ===============
        motor_efforts = None

        humanoid_assets = []
        # load beta into observation ===============
        template_betas = []   # <--- add this
        # load beta into observation ===============

        for i, af in enumerate(asset_file):

            humanoid_asset = self.gym.load_asset(self.sim, asset_root, af, asset_options)

            # load beta into observation ===============
            # load betas for this template
            beta_rel_dir = os.path.dirname(af)                       # e.g. "mjcf/smpl"
            smpl_stem = os.path.splitext(os.path.basename(af))[0]    # "a0f02530_smpl"
            beta_prefix = smpl_stem.rsplit("_", 1)[0]                 # "a0f02530"
            beta_filename = beta_prefix + "_betas.pt"                 # "a0f02530_betas.pt"
            beta_path = os.path.join(asset_root, beta_rel_dir, beta_filename)

            betas = torch.load(beta_path, weights_only=True)
            # here the betas should be torch.Size([1, 10])
            if len(betas.shape) > 1:
                betas = betas[0]

            betas = torch.as_tensor(betas, dtype=torch.float32, device=self.device)
            template_betas.append(betas)
            # load beta into observation ===============

            actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
            curr_motor_efforts = [prop.motor_effort for prop in actuator_props]

            right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "R_Ankle")
            left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "L_Ankle")

            sensor_pose = gymapi.Transform()

            self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
            self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)

            # sensor_count = self.gym.get_asset_force_sensor_count(humanoid_asset)
            # if sensors_per_env is None:
            #     sensors_per_env = sensor_count
            # elif sensor_count != sensors_per_env:
            #     raise ValueError("All humanoid assets must expose the same number of force sensors")

            curr_num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
            curr_num_dof = self.gym.get_asset_dof_count(humanoid_asset)
            curr_num_joints = self.gym.get_asset_joint_count(humanoid_asset)

            if i == 0:
                # the smpl type are of same rigid body and joints, so only take info from the first one

                motor_efforts = curr_motor_efforts

                self.max_motor_effort = max(motor_efforts)
                self.motor_efforts = to_torch(motor_efforts, device=self.device)

                self.torso_index = 0
                self.num_bodies = curr_num_bodies
                self.num_dof = curr_num_dof
                self.num_joints = curr_num_joints

            else:

                assert curr_num_bodies == self.num_bodies, f"diff num_bodies: {curr_num_bodies}, {self.num_bodies}, {af}, {i}"
                assert curr_num_dof == self.num_dof, f"diff num_bodies: {curr_num_dof}, {self.num_dof}"
                assert curr_num_joints == self.num_joints, f"diff num_bodies: {curr_num_joints}, {self.num_joints}"

                if len(curr_motor_efforts) != len(motor_efforts):
                    raise ValueError("All humanoid assets must expose the same number of actuators")
                if not np.allclose(curr_motor_efforts, motor_efforts):
                    raise ValueError("All humanoid assets must share identical actuator effort limits")

            humanoid_assets.append(humanoid_asset)
        
        # load beta into observation ===============
        # torch.Size([64, 10])
        self._template_betas = torch.stack(template_betas, dim=0)    # [T, B]
        # load beta into observation ===============
    
        # multi humanoid template change ===============

        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        # load beta into observation ===============
        # allocate per-env betas for smpl
        beta_dim = self._template_betas.shape[1]
        # torch.Size([2, 10]) [number_actors, betas]
        self._betas_env = torch.zeros(self.num_envs, beta_dim, device=self.device)
        # ---- debug: detect non-finite physics state ----
        self._template_ids_env = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # ---- debug: detect non-finite physics state ----
        # load beta into observation ===============

        # ---- 1211 actions new: morphology-aware root heights ---
        self._base_char_height = self.cfg["env"].get("base_char_height", 0.89)
        self._spawn_height_margin = self.cfg["env"].get("spawn_height_margin", 0.05)
        self._safe_root_heights = torch.zeros(self.num_envs, device=self.device)
        # ---- 1211 actions ------------------------------------------
        
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            # multi humanoid template change ===============
            m = len(humanoid_assets)

            h_asset = humanoid_assets[i % m]

            # load beta into observation ===============
            # assign beta for this env when smpl is used
            template_id = i % m
            self._betas_env[i] = self._template_betas[template_id]
            # ---- debug: detect non-finite physics state ----
            self._template_ids_env[i] = template_id
            # ---- debug: detect non-finite physics state ----
            # load beta into observation ===============

            # ---- 1211 actions new: cache a safe spawn height for this env ---
            self._safe_root_heights[i] = self._compute_safe_root_height(template_id)
            # ---------------------------------------------------

            self._build_env(i, env_ptr, h_asset)
            # multi humanoid template change ===============
            self.envs.append(env_ptr)

        # dof_prop = self.gym.get_actor_dof_properties(self.envs[0], self.humanoid_handles[0])
        # for j in range(self.num_dof):
        #     if dof_prop['lower'][j] > dof_prop['upper'][j]:
        #         self.dof_limits_lower.append(dof_prop['upper'][j])
        #         self.dof_limits_upper.append(dof_prop['lower'][j])
        #     else:
        #         self.dof_limits_lower.append(dof_prop['lower'][j])
        #         self.dof_limits_upper.append(dof_prop['upper'][j])

        # self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        # self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        # collect per-actor dof limits (lower, upper already corrected for swapped bounds)
        dof_lowers_all = []
        dof_uppers_all = []

        for env, handle in zip(self.envs, self.humanoid_handles):
            dof_prop = self.gym.get_actor_dof_properties(env, handle)

            # fix swapped bounds per DOF
            lower = np.minimum(dof_prop['lower'], dof_prop['upper'])
            upper = np.maximum(dof_prop['lower'], dof_prop['upper'])

            dof_lowers_all.append(lower)
            dof_uppers_all.append(upper)

        # shape: [num_actors, num_dof]
        dof_lowers_all = to_torch(np.stack(dof_lowers_all, axis=0), device=self.device)
        dof_uppers_all = to_torch(np.stack(dof_uppers_all, axis=0), device=self.device)

        # global per-DOF limits across all actors
        self.dof_limits_lower, _ = torch.min(dof_lowers_all, dim=0)  # [num_dof]
        self.dof_limits_upper, _ = torch.max(dof_uppers_all, dim=0)  # [num_dof]

        if (self._pd_control):
            self._build_pd_action_offset_scale()

        return
    
    def _build_env(self, env_id, env_ptr, humanoid_asset):
        col_group = env_id
        col_filter = self._get_humanoid_collision_filter()
        segmentation_id = 0

        start_pose = gymapi.Transform()
        # char_h = 0.89

        # ---- 1211 morphology-aware spawn height ---
        # todo we need do this more percisely, findout the exact height for each humanoid
        char_h = float(self._safe_root_heights[env_id]) + 0.1
        # --------------------------------------

        start_pose.p = gymapi.Vec3(*get_axis_params(char_h, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        humanoid_handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", col_group, col_filter, segmentation_id)

        # fetures plugin -------------
        for f in self._features: f.on_humanoid_actor_created(self, env_id, env_ptr)
        # fetures plugin -------------

        # marker logic -------------
        # self._build_target_markers(env_id, env_ptr)
        # marker logic -------------

        # # ---- 1211 actions debug the template ---
        # body_props = self.gym.get_actor_rigid_body_properties(env_ptr, humanoid_handle)

        # masses = [bp.mass for bp in body_props]
        # inertia_eigs = []
        # for bp in body_props:
        #     I = mat33_to_np(bp.inertia).reshape(3, 3)   # inertia is 3×3, row-major
        #     eigs = np.linalg.eigvals(I)
        #     inertia_eigs.append(eigs)

        # ASYM_TOL      = 1e-5   # how non-symmetric is allowed
        # EIG_MIN_TOL   = 1e-6   # min allowed eigenvalue (after symmetrisation)
        # EIG_IMAG_TOL  = 1e-6   # max allowed imaginary part

        # for i, bp in enumerate(body_props):
        #     I = mat33_to_np(bp.inertia).reshape(3, 3)

        #     # symmetry check
        #     asym_norm = np.linalg.norm(I - I.T)

        #     # eigenvalues of raw I (may be complex)
        #     eigs = np.linalg.eigvals(I)
        #     real = eigs.real
        #     imag = eigs.imag

        #     # also check eigenvalues of a symmetrised inertia (physically what we want)
        #     I_sym = 0.5 * (I + I.T)
        #     eigs_sym = np.linalg.eigvalsh(I_sym)  # real by construction

        #     bad_asym = asym_norm > ASYM_TOL
        #     bad_imag = np.max(np.abs(imag)) > EIG_IMAG_TOL
        #     bad_real = np.min(eigs_sym) < EIG_MIN_TOL   # near-zero or negative

        #     if bad_asym or bad_imag or bad_real:
        #         print(f"[WARN] env {env_id} body {i}")
        #         print("I =\n", I)
        #         print("asym_norm:", asym_norm)
        #         print("eigs (raw):", eigs)
        #         print("eigs (sym):", eigs_sym)

        # min_mass   = float(np.min(masses))
        # max_mass   = float(np.max(masses))
        # total_mass = float(np.sum(masses))
        # min_I_eig  = float(np.min([e.min() for e in inertia_eigs]))
        
        # # print("mass min/max/total:", min_mass, max_mass, total_mass)
        # # print("inertia eigen min:", min_I_eig)

        # # ---- Heuristic thresholds (for human-scale characters) ----
        # MIN_SAFE_MASS          = 0.02      # kg, per-link minimum
        # MAX_SAFE_MASS_RATIO    = 200.0     # max_mass / min_mass
        # MIN_SAFE_TOTAL_MASS    = 20.0      # kg
        # MAX_SAFE_TOTAL_MASS    = 120.0     # kg
        # MIN_SAFE_INERTIA_EIG   = 1e-5      # kg·m^2

        # messages = []

        # # Mass checks
        # if min_mass < MIN_SAFE_MASS:
        #     messages.append(f"[WARN] min mass too small: {min_mass:.4f} kg (threshold {MIN_SAFE_MASS} kg)")

        # mass_ratio = max_mass / min_mass if min_mass > 0 else float("inf")
        # if mass_ratio > MAX_SAFE_MASS_RATIO:
        #     messages.append(f"[WARN] mass ratio too large: {mass_ratio:.1f} (threshold {MAX_SAFE_MASS_RATIO})")

        # if not (MIN_SAFE_TOTAL_MASS <= total_mass <= MAX_SAFE_TOTAL_MASS):
        #     messages.append(
        #         f"[WARN] total mass {total_mass:.2f} kg outside [{MIN_SAFE_TOTAL_MASS}, {MAX_SAFE_TOTAL_MASS}] kg"
        #     )

        # # Inertia checks
        # if min_I_eig < MIN_SAFE_INERTIA_EIG:
        #     messages.append(
        #         f"[WARN] minimum inertia eigenvalue too small: {min_I_eig:.3e} "
        #         f"(threshold {MIN_SAFE_INERTIA_EIG:.1e})"
        #     )

        # # Print summary
        # if messages:
        #     print(f"{env_id}: env_id")
        #     print("=== PHYSICAL PROPERTY CHECK: POTENTIAL ISSUES DETECTED ===!!!!!!!!!!!!")
        #     for msg in messages:
        #         print(msg)
        # else:
        #     # print("=== PHYSICAL PROPERTY CHECK: within conservative safety thresholds ===")
        #     pass
        # # ---- 1211 actions debug the template ---

        self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)

        for j in range(self.num_bodies):
            self.gym.set_rigid_body_color(env_ptr, humanoid_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.54, 0.85, 0.2))

        if (self._pd_control):
            dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
            dof_prop["driveMode"] = gymapi.DOF_MODE_POS
            self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)

        self.humanoid_handles.append(humanoid_handle)

        return

    def _build_pd_action_offset_scale(self):
        num_joints = len(self._dof_offsets) - 1
        
        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        for j in range(num_joints):
            dof_offset = self._dof_offsets[j]
            dof_size = self._dof_offsets[j + 1] - self._dof_offsets[j]

            if (dof_size == 3):
                curr_low = lim_low[dof_offset:(dof_offset + dof_size)]
                curr_high = lim_high[dof_offset:(dof_offset + dof_size)]
                curr_low = np.max(np.abs(curr_low))
                curr_high = np.max(np.abs(curr_high))
                curr_scale = max([curr_low, curr_high])
                curr_scale = 1.2 * curr_scale
                curr_scale = min([curr_scale, np.pi])

                lim_low[dof_offset:(dof_offset + dof_size)] = -curr_scale
                lim_high[dof_offset:(dof_offset + dof_size)] = curr_scale
                
                #lim_low[dof_offset:(dof_offset + dof_size)] = -np.pi
                #lim_high[dof_offset:(dof_offset + dof_size)] = np.pi


            elif (dof_size == 1):
                curr_low = lim_low[dof_offset]
                curr_high = lim_high[dof_offset]
                curr_mid = 0.5 * (curr_high + curr_low)
                
                # extend the action range to be a bit beyond the joint limits so that the motors
                # don't lose their strength as they approach the joint limits
                curr_scale = 0.7 * (curr_high - curr_low)
                curr_low = curr_mid - curr_scale
                curr_high = curr_mid + curr_scale

                lim_low[dof_offset] = curr_low
                lim_high[dof_offset] =  curr_high

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)

        return

    def _get_humanoid_collision_filter(self):
        return 0

    def _compute_reward(self, actions):
        self.rew_buf[:] = compute_humanoid_reward(self.obs_buf)
        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_forces, self._contact_body_ids,
                                                   self._rigid_body_pos, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_heights)
        return

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        return

    def _compute_observations(self, env_ids=None):
        obs = self._compute_humanoid_obs(env_ids)

        # load beta into observation ===============
        # append shape betas for smpl assets
        if env_ids is None:
            betas = self._betas_env                        # [num_envs, B]
        else:
            betas = self._betas_env[env_ids]               # [len(env_ids), B]

        # optional: simple normalisation to keep magnitudes modest
        betas = betas / 3.0

        obs = torch.cat([obs, betas], dim=-1)
        # torch.Size([num_envs, 358]) -> torch.Size([num_envs, 368]), 10 betas
        # load beta into observation ===============

        if (env_ids is None):
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs

        return

    def _compute_humanoid_obs(self, env_ids=None):
        if (env_ids is None):
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
        else:
            body_pos = self._rigid_body_pos[env_ids]
            body_rot = self._rigid_body_rot[env_ids]
            body_vel = self._rigid_body_vel[env_ids]
            body_ang_vel = self._rigid_body_ang_vel[env_ids]
        
        obs = compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, self._local_root_obs,
                                                self._root_height_obs)
        return obs

    def _reset_actors(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        return

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()
        if (self._pd_control):
            pd_tar = self._action_to_pd_targets(self.actions)
            pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
        else:
            forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
            force_tensor = gymtorch.unwrap_tensor(forces)
            self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

        return

    def post_physics_step(self):
        self.progress_buf += 1

        self._refresh_sim_tensors()

        # ---- debug: detect non-finite physics state ----
        bad_pos  = ~torch.isfinite(self._rigid_body_pos).all(dim=(1, 2))
        bad_rot  = ~torch.isfinite(self._rigid_body_rot).all(dim=(1, 2))
        bad_vel  = ~torch.isfinite(self._rigid_body_vel).all(dim=(1, 2))
        bad_ang  = ~torch.isfinite(self._rigid_body_ang_vel).all(dim=(1, 2))
        bad_envs = bad_pos | bad_rot | bad_vel | bad_ang

        if bad_envs.any():
            bad_ids = bad_envs.nonzero(as_tuple=False).flatten()
            print("[DEBUG] non-finite physics in envs:", bad_ids.tolist())

            # if you stored template ids, report them:
            if hasattr(self, "_template_ids_env"):
                tmpl_ids = self._template_ids_env[bad_ids].tolist()
                print("[DEBUG] template indices:", tmpl_ids)
                asset_files = self.cfg["env"]["asset"]["assetFileName"]
                for e, t in zip(bad_ids.tolist(), tmpl_ids):
                    print(f"[DEBUG] env {e} uses asset {asset_files[t]}")

            self.reset_buf[bad_ids] = 1

            # 1) reset physics state
            self.reset(env_ids=bad_ids)

            # 2) re-sync sim tensors after reset
            self._refresh_sim_tensors()

            # # 3) rebuild observations for the reset envs
            # self._compute_observations(env_ids=bad_ids)

            # 3) if AMP history exists, wipe it for those envs
            if hasattr(self, "_amp_obs_buf"):
                self._amp_obs_buf[bad_ids] = 0.0
        # -----------------------------------------------

        self._compute_observations()
        self._compute_reward(self.actions)
        self._compute_reset()
        
        self.extras["terminate"] = self._terminate_buf

        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        return

    def render(self, sync_frame_time=False):
        if self.viewer:
            self._update_camera()

        super().render(sync_frame_time)
        return

    def _build_key_body_ids_tensor(self, key_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in key_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_contact_body_ids_tensor(self, contact_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in contact_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _action_to_pd_targets(self, action):
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        return pd_tar

    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()
        
        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0] - 1.0, 
                              self._cam_prev_char_pos[1] - 6.0, 
                              2.0)
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0],
                                 self._cam_prev_char_pos[1],
                                 1.0)

        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()
        
        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0], 
                                  char_root_pos[1] + cam_delta[1], 
                                  cam_pos[2])

        self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos
        return

    def _update_debug_viz(self):
        self.gym.clear_lines(self.viewer)
        return
    
    # marker logic -------------
    # def _load_target_marker_asset(self):

    #     asset_root = self.cfg["env"]["asset"]["assetRoot"]
    #     # You can copy PHC's traj_marker.urdf into <asset_root>/urdf/traj_marker.urdf
    #     marker_asset = "urdf/traj_marker.urdf"

    #     opts = gymapi.AssetOptions()
    #     opts.fix_base_link = True
    #     opts.disable_gravity = True
    #     opts.angular_damping = 0.0
    #     opts.linear_damping = 0.0

    #     self._target_marker_asset = self.gym.load_asset(self.sim, asset_root, marker_asset, opts)

    # def _build_target_markers(self, env_id: int, env_ptr):
    #     # one marker per key body (keeps it cheap and matches MotionLib key_pos)
    #     marker_pose = gymapi.Transform()
    #     marker_pose.p = gymapi.Vec3(0.0, 0.0, 1000.0)  # start hidden

    #     for k in range(self._num_target_markers):
    #         h = self.gym.create_actor(
    #             env_ptr,
    #             self._target_marker_asset,
    #             marker_pose,
    #             f"target_marker_{k}",
    #             env_id,            # col_group
    #             0,                 # col_filter (usually safe for a visual-only URDF)
    #             0
    #         )
    #         # red markers
    #         self.gym.set_rigid_body_color(env_ptr, h, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1.0, 0.0, 0.0))
    #         self._target_marker_handles[env_id, k] = int(h)
    # marker logic -------------

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def dof_to_obs(pose, dof_obs_size, dof_offsets):
    # type: (Tensor, int, List[int]) -> Tensor
    joint_obs_size = 6
    num_joints = len(dof_offsets) - 1

    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device)
    dof_obs_offset = 0

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset:(dof_offset + dof_size)]

        # assume this is a spherical joint
        if (dof_size == 3):
            joint_pose_q = torch_utils.exp_map_to_quat(joint_pose)
        elif (dof_size == 1):
            axis = torch.tensor([0.0, 1.0, 0.0], dtype=joint_pose.dtype, device=pose.device)
            joint_pose_q = quat_from_angle_axis(joint_pose[..., 0], axis)
        else:
            joint_pose_q = None
            assert(False), "Unsupported joint type"

        joint_dof_obs = torch_utils.quat_to_tan_norm(joint_pose_q)
        dof_obs[:, (j * joint_obs_size):((j + 1) * joint_obs_size)] = joint_dof_obs

    assert((num_joints * joint_obs_size) == dof_obs_size)

    return dof_obs

@torch.jit.script
def compute_humanoid_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos,
                                  local_root_obs, root_height_obs, dof_obs_size, dof_offsets):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, int, List[int]) -> Tensor
    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)
    
    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    
    local_root_vel = quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])

    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)

    obs = torch.cat((root_h_obs, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos), dim=-1)
    return obs

@torch.jit.script
def compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    
    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    
    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:] # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])
    
    if (local_root_obs):
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])
    
    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])
    
    obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel), dim=-1)
    return obs


@torch.jit.script
def compute_humanoid_reward(obs_buf):
    # type: (Tensor) -> Tensor
    reward = torch.ones_like(obs_buf[:, 0])
    return reward

@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)
    
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated