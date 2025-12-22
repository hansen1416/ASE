## 3.1 Goal-conditioned motion imitation with AMP (single primitive)

We train a **single goal-conditioned policy** to imitate a large set of motion sequences using reinforcement learning with an **Adversarial Motion Prior (AMP)**. Each episode tracks one reference clip sampled from a dataset (thousands of sequences), while AMP regularizes the controller toward human-like dynamics and improves robustness across diverse styles.

### Markov decision process

At the beginning of an episode, we sample a motion clip (m) and a start index (t_0). The reference trajectory provides target pose and velocities ({\hat{\mathbf{q}}_t,\hat{\dot{\mathbf{q}}}*t}*{t=t_0}^{t_0+T}). The environment state is the simulated humanoid generalized coordinates (\mathbf{q}_t) and velocities (\dot{\mathbf{q}}_t), where (\mathbf{q}_t) includes the root pose and all joint rotations.

We define a goal-conditioned MDP with state
[
\mathbf{s}_t = \big(\mathbf{s}^{p}_t,\ \mathbf{s}^{g}_t\big),
]
where (\mathbf{s}^{p}_t) is proprioception and (\mathbf{s}^{g}_t) encodes short-horizon tracking goals derived from the reference motion.

### Proprioceptive state (\mathbf{s}^{p}_t)

(\mathbf{s}^{p}_t) concatenates simulation features expressed in a **root-aligned local frame** (to reduce dependence on global heading and position), for example:

* root height, root orientation (e.g., 6D/quat), root linear and angular velocity,
* joint rotations and joint angular velocities,
* optional **shape or morphology parameters** (e.g., SMPL (\boldsymbol{\beta})) if training across multiple bodies,
* optional contact indicators or foot heights (useful for stabilizing locomotion-like motions).

All features are normalized by fixed statistics computed from the training set.

### Goal state (\mathbf{s}^{g}_t)

To scale a *single* policy across many heterogeneous clips, the goal representation must be informative but compact. We use a **next-step + lookahead** target encoding:
[
\mathbf{s}^{g}*t = \Big[,\Delta\hat{\mathbf{p}}*{t+1:t+K},\ \Delta\hat{\mathbf{R}}*{t+1:t+K},\ \Delta\hat{\mathbf{v}}*{t+1:t+K},\ \Delta\hat{\boldsymbol{\omega}}_{t+1:t+K},\Big],
]
where (K) is a small horizon (e.g., 1–4). The deltas are measured between reference and simulation in the local root frame, e.g.,

* (\Delta\hat{\mathbf{p}}_{t+k}): link position differences,
* (\Delta\hat{\mathbf{R}}_{t+k}): relative rotation differences (e.g., log-map of (\hat{\mathbf{R}} \mathbf{R}^\top)),
* (\Delta\hat{\mathbf{v}}*{t+k}), (\Delta\hat{\boldsymbol{\omega}}*{t+k}): linear/angular velocity differences.

This goal-conditioning makes a single network act as a **universal tracking controller**: the same policy parameters must fit diverse clips, while the goal provides “what to do next”.

*(If only keypoints are available, (\Delta\hat{\mathbf{R}}) terms are replaced by keypoint position/velocity deltas; the rest of the formulation is unchanged.)*

### Action and low-level control

The policy outputs target joint configurations
[
\mathbf{a}*t = \pi*\theta(\mathbf{s}_t),
]
interpreted as desired joint rotations (or joint-space targets) for a fixed PD controller. Torques are applied as
[
\boldsymbol{\tau}_t
= \mathbf{k}_p \circ (\mathbf{a}_t - \mathbf{q}^{j}_t);-;\mathbf{k}_d \circ \dot{\mathbf{q}}^{j}_t,
]
where (\mathbf{q}^{j}_t) and (\dot{\mathbf{q}}^{j}_t) denote actuated joints, (\mathbf{k}_p,\mathbf{k}_d) are per-joint gains, and (\circ) is elementwise multiplication. This “action-as-target” design encourages stable tracking under noisy or imperfect references and avoids reliance on any auxiliary stabilizing forces.

### Reward function

We combine a dense **imitation reward** with an **AMP reward** and mild regularization:
[
r_t
= w_{im},r^{im}*t;+;w*{amp},r^{amp}*t;-;w*{E},r^{E}*t;-;w*{S},r^{S}_t.
]

**Imitation reward.** We use a multi-term exponential shaping that measures errors over links and joints:
[
r^{im}_t
= w_p,r^{pos}_t + w_r,r^{rot}_t + w_v,r^{vel}*t + w*\omega,r^{ang}_t,
]
with typical terms such as
[
r^{pos}*t=\exp!\Big(-\alpha_p \sum*{\ell}|\mathbf{p}_t^\ell-\hat{\mathbf{p}}_t^\ell|^2\Big),\quad
r^{rot}*t=\exp!\Big(-\alpha_r \sum*{j}|\log(\hat{\mathbf{R}}_t^j(\mathbf{R}_t^j)^\top)|^2\Big),
]
and analogous exponential penalties for linear and angular velocity mismatches. Using exponentials keeps the reward bounded and provides strong gradients near the target while preventing a few outliers from dominating.

**AMP reward.** AMP trains a discriminator (D_\phi(\cdot)) to distinguish **real** motion (reference clips) from **policy-generated** motion. The discriminator consumes a short history of proprioceptive features (\mathbf{h}*t = [\mathbf{s}^{p}*{t-H+1},\ldots,\mathbf{s}^{p}_t]) and outputs a scalar score. We convert this score into a positive reward shaping term (r^{amp}*t=f(D*\phi(\mathbf{h}_t))) (e.g., via a logit-based or exponential mapping), which biases the policy toward realistic transitions even when the imitation target is noisy or ambiguous.

**Regularization.**

* (r^{E}_t): energy/effort penalty (e.g., (|\boldsymbol{\tau}_t|^2) or (|\mathbf{a}_t-\mathbf{q}^j_t|^2)).
* (r^{S}_t): action smoothness penalty (e.g., (|\mathbf{a}*t-\mathbf{a}*{t-1}|^2)) to reduce jitter.

### Episode initialization and termination

**Initialization.** For each sampled clip, we initialize the simulator close to the reference at (t_0) (pose and velocity), optionally with small perturbations (root yaw, small velocity noise) to improve robustness.

**Termination.** The episode terminates if the humanoid enters an unrecoverable configuration for tracking imitation (e.g., root height below a threshold, or large link displacement relative to the reference). To avoid over-penalizing inevitable foot-contact mismatch, termination checks may exclude distal foot bodies (ankle/toe), while still enforcing failure when the torso/root departs significantly.

### Training on thousands of sequences with one primitive

We optimize (\pi_\theta) with PPO and train (D_\phi) concurrently using binary classification (reference vs rollout) on the discriminator input histories. To cover thousands of sequences effectively with a single policy, we use **clip-level sampling control** rather than multiple primitives:

* **uniform sampling** for broad coverage early in training,
* then **prioritized sampling** that increases the probability of clips with low recent tracking success (hard-example mining) while keeping a nonzero uniform mass to prevent mode collapse.

This retains the simplicity of a single primitive while maintaining scalability across large and diverse motion corpora.
