`
python ase/run.py --task HumanoidAMP --cfg_env ase/data/cfg/humanoid_ase_smpl.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file ase/data/motions/0-ACCAD_Female1Running_c3d_C4-Runtowalk1_poses.pkl --headless
`

`
python ase/run.py --task HumanoidAMP --cfg_env ase/data/cfg/humanoid_ase_smpl.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file ase/data/motions/0-ACCAD_MartialArtsWalksTurns_c3d_E15-blockleftmiddle_poses.pkl --headless
`


-----------


## Test

`
python ase/run.py --test --task HumanoidAMP --num_envs 16 --cfg_env ase/data/cfg/humanoid_ase_smpl.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file ase/data/motions/0-ACCAD_Female1Running_c3d_C4-Runtowalk1_poses.pkl --checkpoint /home/hlz/Documents/ase-smpl-run-walk/Humanoid_30-12-08-46/nn/Humanoid_1850.pth
`

`
python ase/run.py --test --task HumanoidAMP --num_envs 16 --cfg_env ase/data/cfg/humanoid_ase_smpl.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file ase/data/motions/0-ACCAD_Female1Running_c3d_C4-Runtowalk1_poses.pkl --checkpoint /home/hlz/Documents/amp-simgle-shape-1212/output/Humanoid_12-12-39-12/nn/Humanoid_1500.pth
`

`
python ase/run.py --test --task HumanoidAMP --num_envs 16 --cfg_env ase/data/cfg/humanoid_ase_smpl.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file ase/data/motions/0-ACCAD_Female1Running_c3d_C4-Runtowalk1_poses.pkl --checkpoint /home/hlz/Documents/amp-52-shape-1211/output/Humanoid_11-15-31-32/nn/Humanoid_3100.pth
`

`
python ase/run.py --test --task HumanoidAMP --num_envs 16 --cfg_env ase/data/cfg/humanoid_ase_smpl.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file ase/data/motions/0-ACCAD_Female1Running_c3d_C4-Runtowalk1_poses.pkl --checkpoint /home/hlz/Documents/amp-multi-shape-1212/output/Humanoid_12-14-15-41/nn/Humanoid_2900.pth
`

`
python ase/run.py --test --task HumanoidAMP --num_envs 16 --cfg_env ase/data/cfg/humanoid_ase_smpl-test.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file ase/data/motions/0-ACCAD_Female1Running_c3d_C4-Runtowalk1_poses.pkl --checkpoint /home/hlz/Documents/amp-multi-shape-1212/output/Humanoid_12-14-15-41/nn/Humanoid_2900.pth
`
