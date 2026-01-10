
- todo we need do calculate the safe height more percisely, findout the exact height for each humanoid


- apply multi shapes loading to target motion loading

- And your reward write-back is correct in the RL sense (reward must end up in self.rew_buf):, why is that?


- observation with motion target of a certain length, refer to PHC,


- The neural network in PHC. phc/learning/mlp.py
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


- chekc all the AMASS dataset with different body shapes, how to make sure there is no penetration at the start?


- do we need to apply multi shape loading to descriminator

- make sure reset also reset velocity.
    root pose and joint pose,
    root linear/angular velocity and joint velocities,
    plus any extra state your controller uses (e.g., PD targets).




--------------------


Check list:

- Are body_pos/body_rot/... and ref_body_pos/ref_body_rot/... aligned over the same rigid-body ordering?
If the motion library’s body order (and which bodies are included) differs from IsaacGym’s rigid_body_tensor order, the reward will be numerically “reasonable” but semantically wrong.
A quick invariant test: if you reset the simulated pose exactly to the reference pose at time t, your imitation reward should jump close to the max (near the weighted sum of your terms).

- How to use the reward:
The contract is:
Your env computes reward into self.rew_buf inside _compute_reward.
The RL runner calls env.step(actions), and step() returns the reward buffer to the algorithm.
PPO/A2C/etc. consumes that returned reward to compute returns/advantages and optimize the policy.
This is explicitly how IsaacGymEnvs-style environments are designed: the RL algorithm calls step() “to retrieve the buffers it needs for training.” 
NVIDIA Developer Forums:
(Modern Isaac Lab describes the same interface shape: step() returns observations, rewards, resets, and extras. 
isaac-sim.github.io)

