## Train

`
python ase/run.py --task HumanoidPHC --cfg_env ase/data/cfg/humanoid_phc.yaml --cfg_train ase/data/cfg/train/rlg/phc_humanoid.yaml --motion_file /home/hlz/datasets/humos_results/ --headless
`
--

## Test

`
python ase/run.py --test --task HumanoidPHC --num_envs 16 --cfg_env ase/data/cfg/humanoid_phc.yaml --cfg_train ase/data/cfg/train/rlg/phc_humanoid.yaml --motion_file /home/hlz/datasets/humos_results/000003_female_1e5a1c90.pkl --checkpoint /home/hlz/Documents/humos-128shape-0226/Humanoid_25-21-06-29/nn/Humanoid_750.pth
`

### note


- load 128 of humanoid, see if their betas covers them all.

- build a class to verify static humanoid models

- /humanoid_phc.py: computes *imitation reward* (compute_imitation_reward) + optional power penalty; writes rew_buf`

- learning/phc_agent.py: overwrites the learning signal by mixing `task_reward_w` and `disc_reward_w` in _combine_rewards().

- ase/data/cfg/train/rlg/phc_humanoid.yaml: shows `task_reward_w`: 0.0 and `disc_reward_w`: 1.0 (so PPO is not optimizing imitation reward unless you changed this).
    - (PHC use task_reward_w: 0.5, disc_reward_w: 0.5)

    - Log raw reward components (reward_raw): pos/rot/vel/ang_vel + power. Is pos/rot high (~0.8-1.0) but vel/ang_vel low? Indicates static bias.

    - Experiment: Scale weights (e.g., increase w_vel to 0.2 in config). Retrain short run; if reward improves, tune.

- utils/motion_lib_humos.py: dataset loading (load_data) and GPU-side motion loading (load_motions).

- env/tasks/humanoid_phc.py: calls _motion_lib.sample_motions() at reset; builds reference at time `t`

    Print/log at runtime (once per epoch):

    _motion_lib._num_unique_motions (dataset size)

    _motion_lib._num_motions (actually loaded motions available for sampling)

    distribution of _sampled_motion_ids over time (should cover all 256 variants)

------


`ase/data/cfg/train/rlg/phc_humanoid.yaml`
define the mlp size and activation function
also which network class to use

The configaritaion is used in network_builders like:
ase/learning/amp_network_builder.py
ase/learning/phc_network_builder.py
