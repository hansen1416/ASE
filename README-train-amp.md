`
python ase/run.py --task HumanoidAMP --cfg_env ase/data/cfg/humanoid_ase_smpl.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file ase/data/motions/0-ACCAD_Female1Running_c3d_C4-Runtowalk1_poses.pkl --headless
`

`
python ase/run.py --task HumanoidAMP --cfg_env ase/data/cfg/humanoid_ase_smpl.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file ase/data/motions/0-ACCAD_MartialArtsWalksTurns_c3d_E15-blockleftmiddle_poses.pkl --headless
`


-----------


## Test

``
python ase/run.py --test --task HumanoidAMP --num_envs 16 --cfg_env ase/data/cfg/humanoid_sword_shield.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/RL_Avatar_Atk_2xCombo01_Motion.npy --checkpoint /home/hlz/Documents/amp-sword/Humanoid_29-16-23-17/nn/Humanoid.pth
``


``
python ase/run.py --test --task HumanoidAMP --num_envs 16 --cfg_env ase/data/cfg/humanoid_ase_smpl.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file ase/data/motions/0-ACCAD_MartialArtsWalksTurns_c3d_E15-blockleftmiddle_poses.pkl --checkpoint /home/hlz/Documents/amp_smpl/Humanoid_29-16-36-43/nn/Humanoid.pth
``
