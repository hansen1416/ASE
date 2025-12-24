- before each call, the corresponding rows in self._root_states for those actor indices already contain the desired reset states.   where we did this?

- If that’s your case, the proper fix is to trigger reset when vis_motion_times >= motion_length (instead of waiting for max_episode_length). why is that, I think the max_episode is always gonna be the motion_length, did phc do this?