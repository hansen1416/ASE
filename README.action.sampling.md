What you have is **not** “ε-greedy in the DQN sense.” It is a **mixture of two action generators** in the same rollout batch:

* **stochastic action**: sample from the policy distribution (a \sim \pi_\theta(\cdot\mid s)) (in your code: `res_dict['actions']`)
* **deterministic action**: take the distribution mean (a=\mu_\theta(s)) (in your code: `res_dict['mus']`)

### 1) What `rand_action_mask` does

```python
rand_action_mask = torch.bernoulli(rand_action_probs)  # 1 with prob p_i
det_action_mask = rand_action_mask == 0.0
res_dict['actions'][det_action_mask] = res_dict['mus'][det_action_mask]
res_dict['rand_action_mask'] = rand_action_mask
```

Per environment (i), you draw a Bernoulli random variable:
[
m_i \sim \mathrm{Bernoulli}(p_i).
]

* If (m_i = 1): keep the sampled action (a_i \sim \mathcal{N}(\mu_i, \sigma_i)).
* If (m_i = 0): override it with the mean action (a_i := \mu_i).

So each rollout step contains a **blend** of “noisy” and “smooth” actions across envs.

### 2) What `_build_rand_action_probs()` is scheduling

```python
self._rand_action_probs = 1.0 - exp(10*(env_id/(N-1) - 1.0))
# env 0 -> ~1.0
# env N-1 -> 0.0
```

It sets a **fixed per-env probability** (p_i) (not time-dependent) based on env index:

* env 0: (p_0 \approx 1) → almost always stochastic
* env N−1: (p_{N-1} = 0) → always deterministic (always (\mu))
* middle envs: smoothly interpolated

And if `enable_eps_greedy` is **disabled**, it forces (p_i = 1) for all envs → fully stochastic rollouts.

### 3) Why do this at all?

There are two motivations (both relevant in AMP):

1. **Smoother trajectories for the discriminator**
   AMP’s discriminator can learn “shortcut features” (e.g., jitter/noise) to distinguish agent vs demo. If every agent trajectory is highly noisy because of sampling, the discriminator can overfit to that noise rather than true motion quality. Mixing in deterministic (\mu)-actions produces smoother clips, making the discriminator focus more on **kinematics/style** than “stochastic jitter.”

2. **Keep exploration for PPO**
   You still need stochastic actions to explore and to provide non-degenerate policy gradients for improving (\mu,\sigma). The stochastic envs provide that.

So the rollout batch is intentionally diversified:

* some envs generate “clean imitation” data (good for discriminator training),
* some envs generate “exploratory” data (good for policy learning).

### 4) The subtle (important) consequence: loss masking

In your ASE `calc_gradients`, you weight actor/entropy/bounds losses by `rand_action_mask` (only stochastic-action envs contribute strongly). This avoids training PPO on transitions where the action was **manually overwritten** to (\mu), which would otherwise break the on-policy log-prob logic.

Intuition:

* For (m=1): action truly came from (\pi_\theta), so PPO loss is valid.
* For (m=0): action was forced to (\mu), not sampled, so treating it as a sample from (\pi_\theta) can bias gradients unless handled carefully. Masking is the simplest safe approach.

### 5) Should you keep it when scaling to thousands of motions?

* If you see discriminator “cheating” on noise/jitter, **keeping this trick can help**.
* If you want the cleanest, standard PPO+AMP baseline, disable it by setting `enable_eps_greedy: false` (which in your code makes everything stochastic), or explicitly set all probs to 1.0.
* If you want “mostly deterministic” to just copy motions: you *can* set all probs to 0 (always (\mu)), but then PPO exploration collapses and learning typically stalls unless you redesign the training objective.

A practical compromise for large-scale imitation is:

* keep it enabled early (helps discriminator quality),
* gradually increase stochastic fraction later (or the reverse), but your current schedule is **by env id**, not by time. If you want a true anneal, change `rand_action_probs` over epochs.

If you want, I can propose a **time-based annealing schedule** that preserves the “smooth-for-disc” benefit early and becomes more standard PPO later.
