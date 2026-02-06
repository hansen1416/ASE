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
        self.motion_lengths = []           # seconds
        self.motion_num_frames = []        # frames
        self.motion_metadata = []          # for debugging: (motion_key, gender_group, beta_key)
        
        self.motion_key_to_length = {}     # str → float
        self.motion_key_to_index = {k: i for i, k in enumerate(self.motion_keys)}

    def num_motions(self) -> int:
        return len(self.motion_keys)

    def get_motion_length(self, idx: int) -> float:
        """Length in seconds for motion at index idx"""
        return float(self.motion_lengths[idx])

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
                data = torch.load(file_path, map_location="cpu")
                
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
                        
                        # Optional: move gender info if needed
                        flat_motion["gender_group"] = gender_group
                        
                        self.motions.append(flat_motion)
                        self.motion_lengths.append(length_sec)
                        self.motion_num_frames.append(num_frames)
                        self.motion_metadata.append((motion_key, gender_group, beta_key))
                        
                        print(f"  Loaded {motion_key}/{gender_group}/{beta_key}: {num_frames} frames, {length_sec:.2f}s")
                        
            except Exception as e:
                print(f"  Error loading {motion_key}: {e}")
                continue

        if not self.motions:
            raise RuntimeError("No valid motion sequences were loaded!")

        # Convert lengths to tensors
        self.motion_lengths = torch.tensor(self.motion_lengths, dtype=torch.float32, device=self.device)
        self.motion_num_frames = torch.tensor(self.motion_num_frames, dtype=torch.long, device=self.device)
        
        print(f"\nSuccessfully loaded {len(self.motions)} individual motion sequences")
        print(f"Total duration: {self.motion_lengths.sum():.2f} seconds")

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

    def get_motion_state(self, motion_ids, motion_times, gender="male", beta_key="0a1ece18"):
        """
        Returns interpolated motion state matching ASE-style keys/shapes.
        Uses the real HUMOS format: pose_body [T, 23, 3], joints_pos [T, 24, 3], etc.
        """
        B = motion_ids.shape[0]
        device = self.device

        root_pos     = torch.zeros(B, 3,       device=device)
        root_rot     = torch.zeros(B, 4,       device=device)  # quat
        dof_pos      = torch.zeros(B, 69,      device=device)  # body 23×3 = 69
        root_vel     = torch.zeros(B, 3,       device=device)
        root_ang_vel = torch.zeros(B, 3,       device=device)
        dof_vel      = torch.zeros(B, 69,      device=device)  # we'll use stored dof_vel
        key_pos      = torch.zeros(B, 24, 3,   device=device)

        for i in range(B):
            mid = motion_ids[i].item()
            motion_key = self.motion_keys[mid]
            t_sec = motion_times[i].item()

            # Find matching variant
            found = False
            flat_motion = None
            for j, meta in enumerate(self.motion_metadata):
                mk, g, bk = meta
                if mk == motion_key and g == gender and bk == beta_key:
                    flat_motion = self.motions[j]
                    found = True
                    break

            if not found:
                # Fallback to first variant of this motion_key
                for j, meta in enumerate(self.motion_metadata):
                    mk, _, _ = meta
                    if mk == motion_key:
                        flat_motion = self.motions[j]
                        break
                if flat_motion is None:
                    continue

            # Extract tensors
            trans       = flat_motion["trans"]          # [T, 3]
            root_orient = flat_motion.get("root_orient")     # [T, 3]
            pose_body   = flat_motion["pose_body"]      # [T, 23, 3]
            joints_pos  = flat_motion.get("joints_pos")      # [T, 24, 3]
            root_vel_t  = flat_motion.get("root_vel")        # [T, 3]
            root_ang_vel_t = flat_motion.get("root_ang_vel") # [T, 3]
            dof_vel_t   = flat_motion.get("dof_vel")         # [T, 23, 3]

            T = trans.shape[0]
            if T < 2:
                root_pos[i] = trans[0]
                if joints_pos is not None:
                    key_pos[i] = joints_pos[0]
                if root_orient is not None:
                    root_rot[i] = torch_utils.axis_angle_to_quaternion(root_orient[0])
                continue

            # Frame blending
            frame_f = t_sec * self.fps
            f0 = int(frame_f)
            f1 = min(f0 + 1, T - 1)
            alpha = frame_f - f0
            alpha = max(0.0, min(1.0, alpha))

            # Root position (translation)
            root_pos[i] = (1 - alpha) * trans[f0] + alpha * trans[f1]

            # Root rotation (axis-angle → quat)
            if root_orient is not None:
                ro0 = root_orient[f0]
                ro1 = root_orient[f1]
                ro_interp = (1 - alpha) * ro0 + alpha * ro1
                root_rot[i] = torch_utils.axis_angle_to_quaternion(ro_interp)
            else:
                root_rot[i] = torch.tensor([0., 0., 0., 1.], device=device)

            # DoF position: body joints only (23 × 3 = 69)
            pb0 = pose_body[f0]     # [23, 3]
            pb1 = pose_body[f1]
            pb_interp = (1 - alpha) * pb0 + alpha * pb1
            dof_pos[i] = pb_interp.reshape(-1)   # → [69]

            # Key positions (global joint locations)
            if joints_pos is not None:
                jp0 = joints_pos[f0]
                jp1 = joints_pos[f1]
                key_pos[i] = (1 - alpha) * jp0 + alpha * jp1

            # Velocities — use precomputed if available
            if root_vel_t is not None:
                rv0 = root_vel_t[f0]
                rv1 = root_vel_t[f1]
                root_vel[i] = (1 - alpha) * rv0 + alpha * rv1

            if root_ang_vel_t is not None:
                ra0 = root_ang_vel_t[f0]
                ra1 = root_ang_vel_t[f1]
                root_ang_vel[i] = (1 - alpha) * ra0 + alpha * ra1

            if dof_vel_t is not None:
                dv0 = dof_vel_t[f0]     # [23, 3]
                dv1 = dof_vel_t[f1]
                dv_interp = (1 - alpha) * dv0 + alpha * dv1
                dof_vel[i] = dv_interp.reshape(-1)   # → [69]

        return {
            "root_pos":     root_pos,
            "root_rot":     root_rot,
            "dof_pos":      dof_pos,
            "root_vel":     root_vel,
            "root_ang_vel": root_ang_vel,
            "dof_vel":      dof_vel,
            "key_pos":      key_pos,
        }