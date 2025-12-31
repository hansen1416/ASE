- move _compute_observations to humanoid_phc, so we can move `_compute_task_obs_v7` to humanoid_phc, 
then in `_compute_task_obs_v7` we can pass `key_pos` to red marker hooks

- merge humanoid.py and humanoid_phc.py, HumanoidPHC will inherit BaseTask directly



- can we merge the marker logic in reset env and reset actor, we probably only need key_pos. To do this, we need to first figure out how PHC build its observation, it should use the target motion as part of the observation.

figure out these logic, if 
 def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)

        if (len(self._reset_default_env_ids) > 0):
            self._init_amp_obs_default(self._reset_default_env_ids)

        if (len(self._reset_ref_env_ids) > 0):
            self._init_amp_obs_ref(self._reset_ref_env_ids, self._reset_ref_motion_ids,
                                   self._reset_ref_motion_times)
        
        return

    def _init_amp_obs_default(self, env_ids):
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs
        return

    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times):
        dt = self.dt
        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps - 1])
        motion_times = motion_times.unsqueeze(-1)
        time_steps = -dt * (torch.arange(0, self._num_amp_obs_steps - 1, device=self.device) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, 
                                              dof_pos, dof_vel, key_pos, 
                                              self._local_root_obs, self._root_height_obs, 
                                              self._dof_obs_size, self._dof_offsets)
        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self._hist_amp_obs_buf[env_ids].shape)
        return


- what is the purpose of `def fetch_amp_obs_demo(self, num_samples)` in humanoid_phc.py



- todo we need do calculate the safe height more percisely, findout the exact height for each humanoid



- load motion in multiple shapes






- observation with motion target of a certain length, refer to PHC,


- reward function like PHC


- check if change betas in AMASS would cause foot penetration,






- apply multi shapes loading to target motion loading


- do we need to apply multi shape loading to descriminator


- add target motion to observation space.