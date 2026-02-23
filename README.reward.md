Adjust the scaling constants (k values) and weights (w values) in self.reward_specs to match PHC's softer penalties.

self.reward_specs = cfg["env"].get("reward_specs", {
    "k_pos": 2.0,       # Lower from 100; PHC uses ~2 for positions
    "k_rot": 0.2,       # Lower from 10; PHC uses ~0.2 for rotations
    "k_vel": 0.1,       # Keep or slightly adjust; PHC uses ~0.1
    "k_ang_vel": 0.1,   # Keep or slightly adjust; PHC uses ~0.1
    "w_pos": 0.5,       # PHC: 0.5 for positions
    "w_rot": 0.3,       # PHC: 0.3 for rotations (or 0.2 if adding end-effector term)
    "w_vel": 0.15,      # PHC: 0.15 for velocities
    "w_ang_vel": 0.05   # PHC: 0.05 for angular velocities
})