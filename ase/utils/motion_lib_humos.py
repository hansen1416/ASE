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
        Returns interpolated motion state for the given motion_ids and times.
        Selects the specific gender + beta_key variant for each motion_key.

        Args:
            motion_ids:     torch.LongTensor [B], indices into self.motion_keys
            motion_times:   torch.Tensor [B], time in seconds
            gender:         str, one of "male", "neutral", "female"
            beta_key:       str, e.g. "0a1ece18"  (same for all envs in this call)

        Returns:
            dict with keys:
                "root_pos"      [B, 3]
                "root_rot"      [B, 4]     (quaternion)
                "dof_pos"       [B, 69]    (usually root 6 + body 63)
                "root_vel"      [B, 3]
                "root_ang_vel"  [B, 3]
                "dof_vel"       [B, 69]
                "key_pos"       [B, 24, 3]   global joint positions
        """
        B = motion_ids.shape[0]
        device = self.device

        # Prepare output tensors
        root_pos     = torch.zeros(B, 3,      device=device)
        root_rot     = torch.zeros(B, 4,      device=device)  # quat
        dof_pos      = torch.zeros(B, 69,     device=device)
        root_vel     = torch.zeros(B, 3,      device=device)
        root_ang_vel = torch.zeros(B, 3,      device=device)
        dof_vel      = torch.zeros(B, 69,     device=device)
        key_pos      = torch.zeros(B, 24, 3,  device=device)

        for i in range(B):
            mid = motion_ids[i].item()
            motion_key = self.motion_keys[mid]
            t = motion_times[i].item()

            # Find the flat_motion that matches motion_key + gender + beta_key
            found = False
            flat_motion = None
            for j, meta in enumerate(self.motion_metadata):
                mk, g, bk = meta
                if mk == motion_key and g == gender and bk == beta_key:
                    flat_motion = self.motions[j]
                    found = True
                    break

            if not found:
                print(f"Warning: variant not found for {motion_key} / {gender} / {beta_key} → using first available")
                # Fallback: use the first one for this motion_key
                for j, meta in enumerate(self.motion_metadata):
                    mk, _, _ = meta
                    if mk == motion_key:
                        flat_motion = self.motions[j]
                        break
                if flat_motion is None:
                    continue

            # Get data
            trans       = flat_motion["trans"]        # [T, 3]
            root_orient = flat_motion.get("root_orient")   # [T, 3] or None
            pose_body   = flat_motion["pose_body"]    # [T, 63]
            joints_pos  = flat_motion.get("joints_pos")    # [T, 24, 3] or None

            num_frames = trans.shape[0]
            if num_frames < 2:
                # degenerate case
                root_pos[i] = trans[0]
                if joints_pos is not None:
                    key_pos[i] = joints_pos[0]
                if root_orient is not None:
                    root_rot[i] = torch_utils.axis_angle_to_quaternion(root_orient[0])
                continue

            # Compute frame indices and blend factor
            frame_float = t / self.dt
            f0 = int(frame_float)
            f1 = min(f0 + 1, num_frames - 1)
            alpha = frame_float - f0
            alpha = max(0.0, min(1.0, alpha))

            # Interpolate
            root_pos[i] = (1 - alpha) * trans[f0] + alpha * trans[f1]

            if root_orient is not None:
                ro0 = root_orient[f0]
                ro1 = root_orient[f1]
                ro_interp = (1 - alpha) * ro0 + alpha * ro1
                root_rot[i] = torch_utils.axis_angle_to_quaternion(ro_interp)
            else:
                root_rot[i] = torch.tensor([0., 0., 0., 1.], device=device)  # identity

            # dof_pos: classic ASE often uses root 6DoF (exp map 3 + trans? but usually just body)
            # Here we do simple concatenation: assume 69 = 6 (root) + 63 (body)
            # But HUMOS doesn't have root translation in dof → we fake simple version
            pb0 = pose_body[f0]   # [63]
            pb1 = pose_body[f1]
            pb_interp = (1 - alpha) * pb0 + alpha * pb1

            # For now: pad with zeros for root 6DoF (common hack for SMPL-based)
            dof_pos[i, :6]  = 0.0
            dof_pos[i, 6:]   = pb_interp   # 63 values

            # key_pos = joints_pos (most useful for visualization / feet contact)
            if joints_pos is not None:
                jp0 = joints_pos[f0]
                jp1 = joints_pos[f1]
                key_pos[i] = (1 - alpha) * jp0 + alpha * jp1
            else:
                # fallback: just repeat root pos (very approximate)
                key_pos[i] = root_pos[i].unsqueeze(0).expand(24, 3)

            # Velocities → placeholder (zero for now)
            # Later: can compute finite difference if needed

        return {
            "root_pos":     root_pos,
            "root_rot":     root_rot,
            "dof_pos":      dof_pos,
            "root_vel":     root_vel,
            "root_ang_vel": root_ang_vel,
            "dof_vel":      dof_vel,
            "key_pos":      key_pos,
        }