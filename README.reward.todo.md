reward-plan:

1. Main imitation tracking reward is `compute_imitation_reward(...)` in `humanoid_im.py`. It computes exponential kernels over body position / rotation / linear velocity / angular velocity tracking errors, then combines them with weights from rwd_specs.

In contrast, the `humanoid.py` in ASE, the default `compute_humanoid_reward(...)` is just a placeholder returning ones. If your HumanoidPHC doesn’t override reward, training will effectively ignore tracking.

Also note: AMP’s discriminator reward is not in the env. It is computed in the agent:

`AMPAgent._calc_disc_rewards(...)` turns discriminator logits into a shaped reward (via a sigmoid → -log(1-prob)), then scales it.

PHC’s “goal-conditioned” observation (difference vectors in a heading-aligned local frame, optionally with futures/lookahead) is also in humanoid_im.py via functions like compute_imitation_observations_v2/v3/v9/... (they explicitly compute local-frame diffs using heading inverse rotation).

todo: Check compute_imitation_reward(...) in humanoid_im.py for the real logic. 4 terms

col: we can just copy the entire function

2. Training on thousands of sequences with one primitive

We optimize (\pi_\theta) with PPO and train (D_\phi) concurrently using binary classification (reference vs rollout) on the discriminator input histories. To cover thousands of sequences effectively with a single policy, we use **clip-level sampling control** rather than multiple primitives:

* **uniform sampling** for broad coverage early in training,
* then **prioritized sampling** that increases the probability of clips with low recent tracking success (hard-example mining) while keeping a nonzero uniform mass to prevent mode collapse.

This retains the simplicity of a single primitive while maintaining scalability across large and diverse motion corpora.

todo: this should be start from a random frame in target motion. 

3. Recovery Rewards; Cycle Consistency; Hard Resets. 

todo: We only need Hard Resets

col: check the reset logic in ASE

5. AMP discriminator reward shaping (the “adversarial prior” reward)

Where: phc/learning/amp_agent.py 
DeepWiki

What it does: Produces an auxiliary reward from a discriminator that distinguishes agent motion vs reference/demo motion (AMP).

todo: check the phc repo for realted logic

6. 3) How rewards are consumed by RL (returns/advantages, PPO losses)

Where: phc/learning/common_agent.py 
DeepWiki

What it does: Implements PPO-style learning machinery that consumes per-step rewards to compute:

Returns / value targets

GAE advantages

Policy/value losses with clipping

DeepWiki explicitly frames PHC as PPO-on-A2C with GAE and clipping. 
DeepWiki

Even though this file isn’t a “reward function,” it is where reward signals become gradients.

todo: check the phc repo for realted logic


7. Imitation training + evaluation plumbing (reward-adjacent)

Where: phc/learning/im_amp.py 
DeepWiki

What it does: Handles imitation-specific training/eval bookkeeping (success rate, MPJPE variants, velocity/acceleration errors). These metrics don’t define reward, but they are used to judge whether the reward shaping is producing the intended behavior.

todo: check the phc repo for realted logic

7. Random-frame reset only works well if you reset the simulator to a dynamically consistent state for that frame:

root pose and joint pose,

root linear/angular velocity and joint velocities,

plus any extra state your controller uses (e.g., PD targets).

If you reset only poses but not velocities, the humanoid often “explodes/drifts” immediately; zero_out_far may then appear necessary, but it is compensating for a reset mismatch rather than solving imitation.

8. do not need zero_out_far. You can focus on the pure imitation branch (i.e., always use compute_imitation_reward), and treat “far from reference” as a termination / reset condition.

_full_body_reward = True