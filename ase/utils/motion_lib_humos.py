# ase/utils/motion_lib_humos.py
import glob
import os
import numpy as np
import torch
import torch.multiprocessing as mp
import random
from tqdm import tqdm
from easydict import EasyDict
import utils.pytorch3d_transforms as torch_utils
import utils.isaacgym_torch_utils as issac_utils

from poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState

from utils.flags import flags
from utils.motion_lib_base import MotionLibBase, MotionlibMode, compute_motion_dof_vels, FixHeightMode
from smpl_sim.smpllib.smpl_parser import SMPL_Parser, SMPLH_Parser, SMPLX_Parser


class MotionLibHUMOS():
    """
    Load HUMOS .pt results and expose the same motion-state API as MotionLibSMPL by
    building SkeletonMotion objects and relying on MotionLibBase.get_motion_state(...).
    """

    def __init__(self, cfg: EasyDict):
        # super().__init__(motion_lib_cfg=motion_lib_cfg)

        # Optional SMPL parsers (only needed if you want mesh-based height fixing).
        # eg. fix_trans_height
        self.cfg = cfg
        self.device = cfg.device
        self.motion_dir = cfg.motion_dir
        self.motion_keys = cfg.motion_keys

        self.fps = 20.0               # HUMOS typically uses 20 fps
        self.dt = 1.0 / self.fps

        # We'll store loaded data here
        self.motions = []                  # list of flat dicts (one per sequence/variant)

        self.motion_metadata = []          # for debugging: (motion_key, gender_group, beta_key)
        
        self.motion_key_num_frames = {}     # str → float
        self.motion_key_to_length = {}     # str → int

        self.motion_key_to_index = {k: i for i, k in enumerate(self.motion_keys)}

        self._variant_idx = {}          # (motion_key, gender, beta_key) -> int (index into self.motions)

    def num_motions(self) -> int:
        return len(self.motion_keys)

    def get_motion_length(self, motion_key: str) -> float:
        """Length in seconds for motion at index idx"""
        return self.motion_key_to_length[motion_key]
    
    def get_motion_frames(self, motion_key: str) -> int:
        """Length in seconds for motion at index idx"""
        return self.motion_key_num_frames[motion_key]
    
    def _variant_id(self, motion_key: str, gender: str, beta_key: str) -> int:
        try:
            return self._variant_idx[(motion_key, gender, beta_key)]
        except KeyError as e:
            raise KeyError(f"Variant not found: {(motion_key, gender, beta_key)}") from e

    def load_motions(self):
        """
        Load HUMOS .pt files and flatten the nested structure into a list of individual motions.
        Each entry in self.motions is a flat dict with keys like: trans, pose_body, root_orient, betas, ...

        humos result is a dict with top-level keys like:
        "male", "neutral", "female", "text"   ← gender-like groups or categories

        Inside each gender key (except "text"):
          another dict with beta_keys (random strings like "00c972db", "a1b2c3d4", ...)
            inside each beta_key: dict with
              "betas"         → shape e.g. [T, 10]   or possibly batched differently
              "gender"        → shape e.g. [T, 1] or scalar per motion
              "root_orient"   → [T, 3]     axis-angle
              "pose_body"     → [T, 63]    flattened axis-angle (21 joints × 3)
              "trans"         → [T, 3]
              "offset_height" → scalar or [T]
              "joints_pos"    → [T, 24, 3]   ← global joint positions (very useful!)
              possibly others like velocities if computed

        """
        print(f"Loading {len(self.motion_keys)} HUMOS motion file(s) from {self.motion_dir}")
       
        for motion_key in tqdm(self.motion_keys, desc="Loading HUMOS files"):
            file_path = os.path.join(self.motion_dir, f"{motion_key}.pt")
            
            if not os.path.isfile(file_path):
                print(f"  → Skipping missing file: {file_path}")
                continue
                
            try:
                data = torch.load(file_path, map_location=self.device)
                
                # data is dict with keys like "male", "neutral", "female", "text"
                for gender_group, group_content in data.items():
                    if gender_group == "text":
                        continue
                        
                    # group_content is dict with beta_keys (random strings)
                    for beta_key, seq_dict in group_content.items():
                        # seq_dict should have "trans", "pose_body", etc.
                        if "trans" not in seq_dict or "pose_body" not in seq_dict:
                            print(f"  → Skipping incomplete sequence: {motion_key}/{gender_group}/{beta_key}")
                            continue
                        
                        # Assume time is dim=0
                        num_frames = seq_dict["trans"].shape[0]
                        length_sec = (num_frames - 1) * self.dt
                        
                        if self.cfg.min_length > 0 and length_sec < self.cfg.min_length:
                            print(f"  → Skipping short seq {motion_key}/{gender_group}/{beta_key}: {length_sec:.2f}s")
                            continue

                        # Inside the beta_key loop, after computing length_sec
                        if motion_key not in self.motion_key_to_length:
                            self.motion_key_to_length[motion_key] = length_sec

                        if motion_key not in self.motion_key_num_frames:
                            self.motion_key_num_frames[motion_key] = num_frames
                        # or assert they are the same if you want to be strict
                        
                        # Store flat dict
                        flat_motion = {
                            "trans":        seq_dict["trans"],
                            "root_orient":  seq_dict.get("root_orient", None),
                            "pose_body":    seq_dict["pose_body"],
                            "betas":        seq_dict.get("betas", None),       # may be constant or per-frame
                            "joints_pos":   seq_dict.get("joints_pos", None),
                            "offset_height":seq_dict.get("offset_height", 0.0),
                            # add more keys if present (root_vel, etc.)
                        }

                        variant_id = len(self.motions)
                        key = (motion_key, gender_group, beta_key)
                        if key in self._variant_idx:
                            raise RuntimeError(f"Duplicate variant key: {key}")

                        self._variant_idx[key] = variant_id 
                        
                        self.motions.append(flat_motion)
                        self.motion_metadata.append((motion_key, gender_group, beta_key))
                        
                        print(f"  Loaded {motion_key}/{gender_group}/{beta_key}: {num_frames} frames, {length_sec:.2f}s")
                        
            except Exception as e:
                print(f"  Error loading {motion_key}: {e}")
                continue

        if not self.motions:
            raise RuntimeError("No valid motion sequences were loaded!")
        
        print(f"\nSuccessfully loaded {len(self.motions)} individual motion sequences")
        print(f"Total duration: {sum(self.motion_key_to_length.values()):.2f} seconds")

    def sample_motions(self, n: int) -> torch.Tensor:
        """
        Sample n motion indices uniformly at random (with replacement).
        
        Returns:
            torch.LongTensor of shape [n], values ∈ [0, self.num_motions()-1]
            Placed on self.device (usually cuda)
        """
        if self.num_motions() == 0:
            raise RuntimeError("No motions loaded. Call load_motions() first.")
            
        # torch.randint is fast and clean
        indices = torch.randint(
            low=0,
            high=self.num_motions(),
            size=(n,),
            device=self.device,
            dtype=torch.long
        )
        return indices

    def sample_time(
        self,
        motion_ids: torch.Tensor,
        truncate_time: float = 0.0
    ) -> torch.Tensor:
        """
        Sample random time ∈ [0, length - truncate_time] for each motion id.
        motion_ids are indices into self.motion_keys.
        """
        n = motion_ids.shape[0]
        lengths = torch.zeros(n, device=self.device)

        for i, mid in enumerate(motion_ids):
            key = self.motion_keys[mid.item()]
            lengths[i] = self.motion_key_to_length[key]

        effective_lengths = torch.clamp(lengths - truncate_time, min=0.01)
        phases = torch.rand(n, device=self.device)
        times = phases * effective_lengths

        return times
    

    def _calc_frame_blend(self, time, len, num_frames, dt):
        time = time.clone()
        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)  # clip time to be within motion length.
        time[time < 0] = 0

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1).long()
        blend = torch.clip((time - frame_idx0 * dt) / dt, 0.0, 1.0) # clip blend to be within 0 and 1
        
        return frame_idx0, frame_idx1, blend

    def get_motion_state(self, motion_ids, motion_times, gender="male", beta_key="0a1ece18"):
        """
        Returns interpolated motion state matching ASE-style keys/shapes.
        Uses the real HUMOS format: pose_body [T, 23, 3], joints_pos [T, 24, 3], etc.
        """
        B = motion_ids.shape[0]

        motion_lens = torch.zeros(B, device=self.device)
        motion_frames = torch.zeros(B, device=self.device)

        for i in range(B):
            mid = motion_ids[i].item()
            motion_key = self.motion_keys[mid]

            motion_lens[i] = self.motion_key_to_length[motion_key]
            motion_frames[i] = self.motion_key_num_frames[motion_key]


        f0, f1, blend = self._calc_frame_blend(motion_times, motion_lens,  motion_frames, self.dt)

        root_pos     = torch.zeros(B, 3,       device=self.device)
        root_rot     = torch.zeros(B, 4,       device=self.device)  # quat
        dof_pos      = torch.zeros(B, 69,      device=self.device)  # body 23×3 = 69
        root_vel     = torch.zeros(B, 3,       device=self.device)
        root_ang_vel = torch.zeros(B, 3,       device=self.device)
        dof_vel      = torch.zeros(B, 69,      device=self.device)  # we'll use stored dof_vel
        key_pos      = torch.zeros(B, 24, 3,   device=self.device)

        rg_pos = torch.zeros(B, 24, 3,   device=self.device)
        rb_rot = torch.zeros(B, 24, 4,   device=self.device)
        body_vel = torch.zeros([B, 24, 3], device=self.device)
        body_ang_vel = torch.zeros([B, 24, 3], device=self.device)

        for i in range(B):
            mid = motion_ids[i].item()
            motion_key = self.motion_keys[mid]
            t_sec = motion_times[i].item()

            # Find matching variant
            variant_id = self._variant_id(motion_key, gender, beta_key)

            flat_motion = self.motions[variant_id]

            # Extract tensors
            trans       = flat_motion["trans"]          # [T, 3]
            root_orient = flat_motion.get("root_orient")     # [T, 3]
            pose_body   = flat_motion["pose_body"]      # [T, 23, 3]
            joints_pos  = flat_motion.get("joints_pos")      # [T, 24, 3]
            root_vel_t  = flat_motion.get("root_vel")        # [T, 3]
            root_ang_vel_t = flat_motion.get("root_ang_vel") # [T, 3]
            dof_vel_t   = flat_motion.get("dof_vel")         # [T, 23, 3]

            # todo, we need to do the interpolation.
            p0 = trans[f0]          # [3]
            p1 = trans[f1]          # [3]
            trans_t = (1.0 - blend) * p0 + blend * p1

            # root_orient: [T,3] axis-angle
            aa0 = root_orient[f0]        # [3]
            aa1 = root_orient[f1]        # [3]

            q0 = torch_utils.axis_angle_to_quaternion(aa0.unsqueeze(0)).squeeze(0)  # [4]
            q1 = torch_utils.axis_angle_to_quaternion(aa1.unsqueeze(0)).squeeze(0)  # [4]
            q_root = issac_utils.slerp(q0, q1, torch.unsqueeze(blend, axis=-1))  # [4]
            # root_rot = q_root

            print(aa0)
            print(q_root)
            print(aa1)
            exit()

            # pose_body: [T,23,3] axis-angle
            pb0 = pose_body[f0]                      # [23,3]
            pb1 = pose_body[f1]                      # [23,3]

            q0 = torch_utils.exp_map_to_quat(pb0.reshape(-1,3)).reshape(23,4)  # [23,4]
            q1 = torch_utils.exp_map_to_quat(pb1.reshape(-1,3)).reshape(23,4)  # [23,4]

            blend_t = torch.full((23,1), blend, device=q0.device, dtype=q0.dtype)  # broadcast
            qj = torch_utils.slerp(q0, q1, blend_t)                                 # [23,4]

            pose_body_t = torch_utils.quat_to_exp_map(qj).reshape(23,3)             # back to axis-angle (exp-map)
            # dof_pos = pose_body_t.reshape(-1)  # [69]



        return {
            "root_pos":     root_pos,
            "root_rot":     root_rot,
            "dof_pos":      dof_pos,
            "root_vel":     root_vel,
            "root_ang_vel": root_ang_vel,
            "dof_vel":      dof_vel,
            "key_pos":      key_pos,

            # Added to match MotionLibBase style
            "rg_pos":       rg_pos,
            "rb_rot":       rb_rot,
            "body_vel":     body_vel,
            "body_ang_vel": body_ang_vel,
        }