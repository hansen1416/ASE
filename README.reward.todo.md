reward-plan:

1. Main imitation tracking reward is `compute_imitation_reward(...)` in `humanoid_im.py`. It computes exponential kernels over body position / rotation / linear velocity / angular velocity tracking errors, then combines them with weights from rwd_specs.

In contrast, the `humanoid.py` in ASE, the default `compute_humanoid_reward(...)` is just a placeholder returning ones. If your HumanoidPHC doesn’t override reward, training will effectively ignore tracking.

Also note: AMP’s discriminator reward is not in the env. It is computed in the agent:

`AMPAgent._calc_disc_rewards(...)` turns discriminator logits into a shaped reward (via a sigmoid → -log(1-prob)), then scales it.

PHC’s “goal-conditioned” observation (difference vectors in a heading-aligned local frame, optionally with futures/lookahead) is also in humanoid_im.py via functions like compute_imitation_observations_v2/v3/v9/... (they explicitly compute local-frame diffs using heading inverse rotation).

