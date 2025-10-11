"""
ase/data/motions/amp_humanoid_walk.npy
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69]
tensor([ 7,  3, 22, 17], device='cuda:0')
"""

from enum import Enum
import numpy as np
import os

from isaacgym import gymapi
from isaacgym import gymtorch

import torch
from easydict import EasyDict

from env.tasks.humanoid import Humanoid, dof_to_obs
from utils import gym_util
from utils.motion_lib_smpl import MotionLibSMPL
from poselib.poselib.skeleton.skeleton3d import SkeletonTree

device = 'cuda:0'
motion_file = "ase/data/motions/amass/0-ACCAD_Female1General_c3d_A2-Sway_poses.pkl"
# _dof_body_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
# _dof_offsets = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69]
_key_body_ids = torch.tensor([ 7,  3, 22, 17], device=device)

asset_file = os.path.join(os.path.dirname(__file__), "./data/assets/mjcf/smpl_humanoid.xml")

sk_tree = SkeletonTree.from_mjcf(asset_file)

gender_beta = np.zeros(17)
num_envs = 1

humanoid_shapes = torch.tensor(np.array([gender_beta] * num_envs)).float().to(device)


motion_lib_cfg = EasyDict({
                "motion_file": motion_file,
                "device": torch.device("cpu"),
                "fix_height": 1,
                "min_length": -1,
                "max_length": -1,
                "im_eval": True,
                "multi_thread": False,
                "smpl_type": "smpl",
                "randomrize_heading": True,
                "device": device,
                "min_length": -1, 
                "step_dt": 1/60,
            })



motion_lib = MotionLibSMPL(motion_lib_cfg=motion_lib_cfg)
motion_lib.load_motions(skeleton_trees=[sk_tree], 
                        gender_betas=humanoid_shapes.cpu(), 
                        random_sample=True)

# tensor([0], device='cuda:0')
motion_ids = motion_lib.sample_motions(num_envs)
# tensor([2.9572], device='cuda:0')
motion_times = motion_lib.sample_time(motion_ids, truncate_time=0.0)

results = motion_lib.get_motion_state(motion_ids, motion_times, _key_body_ids)

root_pos = results['root_pos']
root_rot = results['root_rot']
dof_pos = results['dof_pos']
root_vel = results['root_vel']
root_ang_vel = results['root_ang_vel']
dof_vel = results['dof_vel']
key_pos = results['key_pos']

print("root_pos", root_pos.shape)
print("root_rot", root_rot.shape)
print("dof_pos", dof_pos.shape)
print("root_vel", root_vel.shape)
print("root_ang_vel", root_ang_vel.shape)
print("dof_vel", dof_vel.shape)
print("key_pos", key_pos.shape)