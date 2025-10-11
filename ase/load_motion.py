"""
ase/data/motions/amp_humanoid_walk.npy
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69]
tensor([ 7,  3, 22, 17], device='cuda:0')
"""

from enum import Enum
import numpy as np

from isaacgym import gymapi
from isaacgym import gymtorch

import torch

from env.tasks.humanoid import Humanoid, dof_to_obs
from utils import gym_util
from utils.motion_lib import MotionLib

device = 'cuda:0'
motion_file = "ase/data/motions/amp_humanoid_walk.npy"
_dof_body_ids = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
_dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
_key_body_ids = torch.tensor([ 5,  8, 11, 14], device=device)

motion_lib = MotionLib(motion_file=motion_file,
                                     dof_body_ids=_dof_body_ids,
                                     dof_offsets=_dof_offsets,
                                     key_body_ids=_key_body_ids.cpu().numpy(), 
                                     device=device)

num_envs = 1

# tensor([0], device='cuda:0')
motion_ids = motion_lib.sample_motions(num_envs)
# tensor([0.2708], device='cuda:0')
motion_times = motion_lib.sample_time(motion_ids, truncate_time=0.0)

root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = motion_lib.get_motion_state(motion_ids, motion_times)

print("root_pos", root_pos.shape)
print("root_rot", root_rot.shape)
print("dof_pos", dof_pos.shape)
print("root_vel", root_vel.shape)
print("root_ang_vel", root_ang_vel.shape)
print("dof_vel", dof_vel.shape)
print("key_pos", key_pos.shape)