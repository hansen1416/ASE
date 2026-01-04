reward-plan:

1. Main imitation tracking reward is `compute_imitation_reward(...)` in `humanoid_im.py`. It computes exponential kernels over body position / rotation / linear velocity / angular velocity tracking errors, then combines them with weights from rwd_specs.

In contrast, the `humanoid.py` in ASE, the default `compute_humanoid_reward(...)` is just a placeholder returning ones. If your HumanoidPHC doesn’t override reward, training will effectively ignore tracking.

Also note: AMP’s discriminator reward is not in the env. It is computed in the agent:

`AMPAgent._calc_disc_rewards(...)` turns discriminator logits into a shaped reward (via a sigmoid → -log(1-prob)), then scales it.

PHC’s “goal-conditioned” observation (difference vectors in a heading-aligned local frame, optionally with futures/lookahead) is also in humanoid_im.py via functions like compute_imitation_observations_v2/v3/v9/... (they explicitly compute local-frame diffs using heading inverse rotation).

2. Training on thousands of sequences with one primitive

We optimize (\pi_\theta) with PPO and train (D_\phi) concurrently using binary classification (reference vs rollout) on the discriminator input histories. To cover thousands of sequences effectively with a single policy, we use **clip-level sampling control** rather than multiple primitives:

* **uniform sampling** for broad coverage early in training,
* then **prioritized sampling** that increases the probability of clips with low recent tracking success (hard-example mining) while keeping a nonzero uniform mass to prevent mode collapse.

This retains the simplicity of a single primitive while maintaining scalability across large and diverse motion corpora.


