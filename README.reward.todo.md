reward-plan:

1. Main imitation tracking reward is `compute_imitation_reward(...)` in `humanoid_im.py`. It computes exponential kernels over body position / rotation / linear velocity / angular velocity tracking errors, then combines them with weights from rwd_specs.

In contrast, the `humanoid.py` in ASE, the default `compute_humanoid_reward(...)` is just a placeholder returning ones. If your HumanoidPHC doesn’t override reward, training will effectively ignore tracking.

Also note: AMP’s discriminator reward is not in the env. It is computed in the agent:

`AMPAgent._calc_disc_rewards(...)` turns discriminator logits into a shaped reward (via a sigmoid → -log(1-prob)), then scales it.

PHC’s “goal-conditioned” observation (difference vectors in a heading-aligned local frame, optionally with futures/lookahead) is also in humanoid_im.py via functions like compute_imitation_observations_v2/v3/v9/... (they explicitly compute local-frame diffs using heading inverse rotation).

todo: Check compute_imitation_reward(...) in humanoid_im.py for the real logic. 4 terms

col: just copy the entire function, and use the motion res in post_physics_step instead. 
do not need zero_out_far. focus on the pure imitation branch (i.e., always use compute_imitation_reward), 
and treat “far from reference” as a termination / reset condition.
_full_body_reward = True

check if the reward jumps when reset to motion frame



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

--

Hard Resets. 

todo: We only need Hard Resets

col: check the reset logic in ASE, it should be fine, right now we are doing hard reset. Ask gpt to confirm it.
also, how to reset the velocity


8. The neural network in PHC


9. And your reward write-back is correct in the RL sense (reward must end up in self.rew_buf):, why is that?


---------


In case it didn't work out check list:

Are body_pos/body_rot/... and ref_body_pos/ref_body_rot/... aligned over the same rigid-body ordering?
If the motion library’s body order (and which bodies are included) differs from IsaacGym’s rigid_body_tensor order, the reward will be numerically “reasonable” but semantically wrong.

A quick invariant test: if you reset the simulated pose exactly to the reference pose at time t, your imitation reward should jump close to the max (near the weighted sum of your terms).

--

How to use the reward:
The contract is:

Your env computes reward into self.rew_buf inside _compute_reward.

The RL runner calls env.step(actions), and step() returns the reward buffer to the algorithm.

PPO/A2C/etc. consumes that returned reward to compute returns/advantages and optimize the policy.

This is explicitly how IsaacGymEnvs-style environments are designed: the RL algorithm calls step() “to retrieve the buffers it needs for training.” 
NVIDIA Developer Forums

(Modern Isaac Lab describes the same interface shape: step() returns observations, rewards, resets, and extras. 
isaac-sim.github.io
)

--

PHC commonly exposes diagnostics (e.g., per-term reward components) through self.extras[...] so the trainer/logger can record them.

If you want PHC-like visibility in TensorBoard/W&B logs, add:

# in post_physics_step(), after _compute_reward()
self.extras["reward_raw"] = self.reward_raw.detach()
self.extras["reward_total"] = self.rew_buf.detach()

--

Minimal checklist (so PPO actually receives your reward)

self.rew_buf is shape (num_envs,), on the correct device, float32.

You overwrite self.rew_buf[:] every step (not accumulate).

reset_buf is set correctly (done flags), otherwise returns/episodes will be wrong even if rewards are right.