## Train

`
python ase/run.py --task HumanoidPHC --cfg_env ase/data/cfg/humanoid_phc.yaml --cfg_train ase/data/cfg/train/rlg/phc_humanoid.yaml --motion_file /home/hlz/datasets/humos_results/ --headless
`
-----------


## Test

`
python ase/run.py --test --task HumanoidPHC --num_envs 16 --cfg_env ase/data/cfg/humanoid_phc.yaml --cfg_train ase/data/cfg/train/rlg/phc_humanoid.yaml --motion_file /home/hlz/datasets/humos_results/000003_female_1e5a1c90.pkl --checkpoint /home/hlz/Documents/humos-128shape-0226/Humanoid_25-21-06-29/nn/Humanoid_750.pth
`