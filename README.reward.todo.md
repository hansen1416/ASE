reward-plan:

1. Migrated PHC `compute_imitation_reward` in `humanoid_im.py` to ASE, with zero_out_far = False; _full_body_reward = True.
`compute_imitation_reward` has 4 terms: # body position reward; # body rotation reward; # body linear velocity reward; # body angular velocity reward

Verified the reward is maximum when reset the humanoid.



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





10. 
Hyperparameter
Value
Context
PPO Clip ($\epsilon$)
0.2
Standard for stable on-policy updates.6
Learning Rate
$5 \times 10^{-5}$
Conservative rate to prevent PNN column divergence.13
AMP Reward Scale
2.0
High enough to enforce "style" over "shortcuts".15
Tracking Reward Scale
10.0
Primary signal for the imitation objective.8
Batch Size
2048
Balanced for GPU memory and gradient stability.25
PNN Hidden Layers


High capacity for complex skeletal dynamics.4


11. 

`self.rnn_states` is **not** “PHC’s own neural network structure”; it is the **recurrent hidden-state container** that RL-Games passes between the agent and the policy network **only when the policy is RNN-based** (e.g., GRU/LSTM).

### Why does it exist / what does it do?

* In PHC’s `IMAmpAgent.get_action`, the agent passes `rnn_states` into the model and receives updated `rnn_states` back each step (PHC stores it in `self.states`, but the key is still `"rnn_states"`). 
* PHC initializes these states via `model.get_default_rnn_state()` and allocates a `[num_layers, num_envs, hidden_dim]`-shaped tensor bank.
* In `AMPAgent`, the RNN rollout path (`play_steps_rnn`) likewise updates `self.rnn_states` from `res_dict['rnn_states']` and resets them on episode termination.
* If `self.is_rnn == False`, `rnn_states` is effectively unused (typically `None` or an empty structure), and nothing “recurrent” happens.

So: **`self.rnn_states` exists because RL-Games supports recurrent policies**, and the agent must carry per-environment hidden state across timesteps. It does *not* imply PHC uses a fundamentally different “agent-owned” network.

### Does PHC use custom network structure anyway?

Yes—but that’s **orthogonal** to `rnn_states`. PHC’s AMP agent assumes a customized `a2c_network` API that includes discriminator-related heads/methods (e.g., `get_disc_logit_weights()`, discriminator logits in `res_dict`).
That “custom structure” is part of **AMP (policy + discriminator)**, not specifically an RNN feature.

### Should you migrate the network first?

Depends on your goal:

* **Training from scratch in ASE:** usually *no*. You can migrate agent logic first, as long as your ASE model already exposes the same AMP-specific outputs/methods (discriminator logits, etc.). The `rnn_states` plumbing only matters if you enable RNN in the config.

* **Reusing PHC checkpoints / expecting identical behavior:** *yes*. Then you should migrate **network builder + exact model architecture + normalization modules** before expecting restores to work, because weight loading is shape/name sensitive (PHC even guards against mean/std shape mismatch on restore).

### Practical takeaway

Treat `rnn_states` as an **interface contract**:

* If your ASE run is **non-recurrent**, keep it simple: `is_rnn: False`, ignore `rnn_states`.
* If you enable **recurrent policies**, ensure you migrate the *RNN rollout path* (state init/reset, masks) consistently—then `rnn_states` becomes necessary. 






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

9. where they combine the task reward and the amp reward 


--

4) Persist + restore “failure/termination history” to continue curriculum

PHC’s restore() loads a saved “failed_*” artifact and calls something like motion_lib.update_sampling_prob(termination_history) to restore the sampling distribution / termination history across runs. 

im_amp

ASE AMPAgent has no such restore-extension.