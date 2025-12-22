| Aspect                        | PHC (Perpetual Humanoid Control)                                                                                                                                                                    | AMP-style imitation (as you have now)                                                                                                                                     |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Core objective                | **Track a specific reference motion** (per-env target trajectory) robustly over time, including recovery.                                                                                           | **Match a motion distribution/style** via a discriminator; no requirement to follow a particular clip at each step.                                                       |
| Per-step conditioning         | Maintains ((\text{motion_id}, t)) per env; computes **reference state** (x^*(t)) each step; uses **goal deltas** (often + lookahead) in the *main observation*.                                     | Typically observes **proprioception (+ β)**; demo motions mainly feed the discriminator (and sometimes AMP-history obs), not an explicit per-step target in the main obs. |
| Reward definition             | Dominated by **tracking reward**: exponential penalties on pose/velocity/orientation/keypoint errors vs (x^*(t)); often plus **energy/torque** regularization; style reward (if used) is auxiliary. | Dominated by **discriminator reward** (how “real” the motion looks) plus any simple task term; no explicit tracking error to a reference.                                 |
| Motion library usage          | **Online target generation**: MotionLib queried every step (and for future steps if lookahead) to build goal features and rewards.                                                                  | **Demo supply**: MotionLib used mainly to provide **real samples** for discriminator training (and AMP-demo obs buffers).                                                 |
| Time/episode structure        | Clip-aware: advance motion time; reset when **clip ends** (or near end with lookahead margin); fall handling often integrated with tracking progress.                                               | Not clip-indexed by default: episodes are primarily timeouts/fall-based; no notion of “progress along a specific clip” unless you add it.                                 |
| Termination & resets          | Often include **tracking-specific criteria**: fall, excessive deviation, clip end; may keep the agent alive to learn recovery before reset.                                                         | Usually reset on fall/timeout; “deviation from a reference” is not a native termination signal because there is no explicit reference.                                    |
| Training loop extras          | Frequently adds **recovery training / hard-negative mining** (initialize from failures, learn to stand up and continue).                                                                            | Standard AMP loop: alternate policy updates and discriminator updates; no built-in recovery mining unless added.                                                          |
| Observation dimensionality    | Larger: proprio + goal deltas (+ lookahead), sometimes reference keypoints, future root trajectory, etc.                                                                                            | Smaller: proprio (+ β) + optional AMP-history features; no target-conditioning unless you implement it.                                                                   |
| Policy/hyperparameter demands | Typically needs **deeper networks**, careful PPO batch sizing, and reward balancing for stable tracking.                                                                                            | Can work with **lighter networks**; discriminator provides dense gradients even without precise tracking structure.                                                       |


---------------



### Reward: yes (mostly in the task/env code)

In PHC the “tracking-style” reward is implemented as TorchScript kernels inside the imitation task utilities, not inside the policy network.

* **Main imitation tracking reward** is `compute_imitation_reward(...)` in `humanoid_im.py`. It computes exponential kernels over **body position / rotation / linear velocity / angular velocity** tracking errors, then combines them with weights from `rwd_specs`. 
* **Termination/reset** logic for imitation is also there (e.g., `compute_humanoid_im_reset(...)`). 
* In contrast, in your ASE-side `humanoid.py`, the default `compute_humanoid_reward(...)` is just a placeholder returning ones. If your `HumanoidPHC` doesn’t override reward, training will effectively ignore tracking. 

Also note: **AMP’s discriminator reward is *not* in the env**. It is computed in the agent:

* `AMPAgent._calc_disc_rewards(...)` turns discriminator logits into a shaped reward (via a sigmoid → `-log(1-prob)`), then scales it. 

### Neural network structure: no (it lives in rl_games “builder” + YAML)

PHC (and ASE/AMP) follow the rl_games pattern: **task files define obs/reward/reset; network architecture is built by the RL framework**.

* `run.py` registers which **agent/model/network builder** to use for `algo=amp` vs `algo=ase`. For AMP it wires `AMPAgent` + `ModelAMPContinuous` + `AMPBuilder`. 
* The **actual actor–critic MLP and discriminator MLP** are constructed in `amp_network_builder.py` (see `eval_actor`, `eval_critic`, and `_build_disc`). The discriminator MLP sizes/activation come from config (YAML). 

### Bonus: where PHC “goal deltas (+ lookahead)” obs is implemented

PHC’s “goal-conditioned” observation (difference vectors in a heading-aligned local frame, optionally with futures/lookahead) is also in `humanoid_im.py` via functions like `compute_imitation_observations_v2/v3/v9/...` (they explicitly compute local-frame diffs using heading inverse rotation).
