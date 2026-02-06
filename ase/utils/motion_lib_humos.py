# ase/utils/motion_lib_humos.py
import glob
import os
import numpy as np
import torch
import torch.multiprocessing as mp
import random
from tqdm import tqdm
from scipy.spatial.transform import Rotation as sRot

from poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState

from utils.flags import flags
from utils.motion_lib_base import MotionLibBase, MotionlibMode, compute_motion_dof_vels, FixHeightMode
from smpl_sim.smpllib.smpl_parser import SMPL_Parser, SMPLH_Parser, SMPLX_Parser


class MotionLibHUMOS():
    """
    Load HUMOS .pt results and expose the same motion-state API as MotionLibSMPL by
    building SkeletonMotion objects and relying on MotionLibBase.get_motion_state(...).
    """

    def __init__(self, motion_lib_cfg):
        # super().__init__(motion_lib_cfg=motion_lib_cfg)

        # Optional SMPL parsers (only needed if you want mesh-based height fixing).
        # eg. fix_trans_height
        self.mesh_parsers = None

        self.motion_dir = motion_lib_cfg.motion_dir
        self.motion_keys = motion_lib_cfg.motion_keys
        # in HUMOS, “We first subsample raw SMPL-H sequences from AMASS … to 20 fps …
        self.fps = 20

    def num_motions(self):
        return len(self.motion_keys)

    def load_motions(self):
        """
        humos_result: the strucure is {motion_key: {gender_index: ['betas', 'gender', 'root_orient', 'pose_body', 'trans', 'offset_height', 'text'], ....}}

        betas: torch.Size([64, 200, 10])
        gender: torch.Size([64, 200, 1])
        root_orient: torch.Size([64, 200, 3])
        pose_body: torch.Size([64, 200, 63])
        trans: torch.Size([64, 200, 3])
        offset_height: torch.Size([64])
        
        :param self: Description
        """

        motions = {}

        # the motion id string
        for motion_key in self.motion_keys:

            motion_file_path = os.path.join(self.motion_dir, f"{motion_key}.pt")

            if os.path.isfile(motion_file_path):
                # motion format: 
                # betas:        torch.Size([200, 10])
                # gender:       torch.Size([200, 1])
                # root_orient:  torch.Size([200, 3])
                # pose_body:    torch.Size([200, 23, 3])
                # trans:        torch.Size([200, 3])
                # offset_height:torch.Size([])
                # joints_pos:   torch.Size([200, 24, 3])
                # root_vel:     torch.Size([200, 3])
                # root_ang_vel: torch.Size([200, 3])
                # dof_vel:      torch.Size([200, 23, 3])
                motion = torch.load(motion_file_path, map_location="cpu")

                motions.append(motion)

            else:
                print(f"warning! motion file does not exists {motion_file_path}")
        

    def get_motion_state(self, motion_ids, motion_times, gender="male", beta_key="00c972db"):

        pass

    
