1. Introduce betas into the observations
    - allow env load multiple humanoid models, refer to code in `# multi humanoid template change ===============` in ase/env/tasks/humanoid_amp.py and ase/env/tasks/humanoid.py
    - load beta into observation , refer to code in `# load beta into observation ===============` in ase/env/tasks/humanoid_amp.py and ase/env/tasks/humanoid.py
    - code in `---- 1211 actions` is the attempt to fix unstable humanoid models.
2. Generate multiple betas
    - why 64 body shapes, refer to notes in ./betas
    - use `"real_weight": False, ` to fix some of the unstable humanoid.
    - tuning 
    ```
        if bone.name in ["Torso", "Chest", "Spine"]:
            seperation = 0.6
        else:
            seperation = 0.2
    ```
    in `smpl_sim/smpllib/skeleton_local.py` to fix penetration, attempt to fix them all. but there is still something wrong, try to adjust `seperation` and `capsule size`
    at this stage, we have 64 stable ones.

3. deffierence motion load stratergy.

4. new reward function for ASE, take the target motion into consideration.

    - use global `motion_times`, in reset_actor and `_compute_observations`. extract `_compute_task_obs_v7` to top level, calculate obs, key_pos, and pass them to reset_actor and _compute_observations. so we don't have to calculate motion state in `reset_actor`.

5. let target motion load multiple body shapes.

- red marker logic

- move _compute_observations to humanoid_phc, so we can move `_compute_task_obs_v7` to humanoid_phc, 
then in `_compute_task_obs_v7` we can pass `key_pos` to red marker hooks

- merge humanoid.py and humanoid_phc.py, HumanoidPHC will inherit BaseTask directly.

- can we merge the marker logic in reset env and reset actor, we probably only need key_pos. To do this, we need to first figure out how PHC build its observation, it should use the target motion as part of the observation.


- what is the purpose of `def fetch_amp_obs_demo(self, num_samples)` in humanoid_phc.py. Build the descriminator reward


- add target motion to observation space.

- Migrated PHC `compute_imitation_reward` in `humanoid_im.py` to ASE, with zero_out_far = False; _full_body_reward = True.
`compute_imitation_reward` has 4 terms: # body position reward; # body rotation reward; # body linear velocity reward; # body angular velocity reward.
Verified the reward is maximum when reset the humanoid.

- it's always Hard Resets to a frame in the target motion, no fail recover


