## Original ASE config, load PHC pickle

`
python ase/run.py --task HumanoidViewMotion --num_envs 2 --cfg_env ase/data/cfg/humanoid_ase_smpl.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file /home/hlz/datasets/amass-pkls/0-ACCAD_MartialArtsWalksTurns_c3d_E15-blockleftmiddle_poses.pkl
`

## PHC config, load PHC pickle

`
python ase/run.py --task HumanoidViewMotion --num_envs 4 --cfg_env ase/data/cfg/humanoid_phc.yaml --cfg_train ase/data/cfg/train/rlg/phc_humanoid.yaml --motion_file /home/hlz/datasets/amass-pkls/0-ACCAD_Female1Running_c3d_C4-Runtowalk1_poses.pkl
`

## PHC config, load HUMOS pickle

`
python ase/run.py --task HumanoidViewMotion --num_envs 4 --cfg_env ase/data/cfg/humanoid_phc.yaml --cfg_train ase/data/cfg/train/rlg/phc_humanoid.yaml --motion_file /home/hlz/datasets/humos_results/a_person_stumbles_to_the_female_fcc491cd.pkl
`

`
/home/hlz/datasets/humos_results/with_their_left_hand_th_neutral_74fc526e.pkl
/home/hlz/datasets/humos_results/a_person_squats_down_and_neutral_aef91182.pkl
/home/hlz/datasets/humos_results/person_waves_with_their_neutral_15c4d2e9.pkl
/home/hlz/datasets/humos_results/person_walks_up_and_squa_neutral_5636a12a.pkl
/home/hlz/datasets/humos_results/a_person_pats_the_top_of_neutral_31f56211.pkl
`