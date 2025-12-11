# Action list for debugging RL physics / AMP pipeline

---

## 2. Asset loading: inertia and self-collision

**Where**

* `env/tasks/humanoid.py`, in `_create_envs` (right before calling `gym.load_asset` and after creating `gymapi.AssetOptions()`).

**What**

* Enable robust convex decomposition and inertia:

  * `asset_options.vhacd_enabled = True`
  * `asset_options.override_com = True`
  * `asset_options.override_inertia = True`
* If needed, relax self-collision between adjacent links (e.g. pelvis–thigh) via collision groups/masks.
* Goal: ensure inertia and collision shapes stay consistent when SMPL geometry/scale changes.

---

## 3. Morphology-aware spawn height and safe resets

**Where**

* `env/tasks/humanoid.py`:

  * `_build_env` (initial root pose; currently uses a single `char_h` for all envs).
  * `_reset_envs` / `_reset_actors`.
* `env/tasks/humanoid_amp.py`:

  * Any overrides of reset logic for AMP.

**What**

* Replace fixed spawn height (e.g. `char_h = 0.89`) with per-env safe height:

  * Compute a per-env scale or height factor (e.g. from SMPL betas or template scale).
  * Define `safe_height = base_height * scale + margin` (e.g. `0.9 * scale + 0.05`).
  * Set the root state z coordinate to `safe_height` before writing back to the actor state tensor.
* Apply the same logic on resets (not only at initial creation).
* Goal: avoid deep ground penetration for tall/short morphologies at reset.

---

## 4. NaN-safe, clamped observations (env side)

**Where**

* `env/tasks/humanoid.py`, in `_compute_observations` right before assigning to `self.obs_buf`.

**What**

* After constructing the full observation (including betas), add:

  * **Clamping**: elementwise clamp, e.g. `[-100, 100]`.
  * **NaN/Inf cleaning**: apply `torch.nan_to_num` (or equivalent) on the observation tensor.
* Then write the sanitized tensor into `self.obs_buf`.
* Goal: guarantee that no non-finite or extreme values leave the environment.

---

## 5. AMP observation safety (env side)

**Where**

* `env/tasks/humanoid_amp.py`, in `HumanoidAMP.post_physics_step`:

  * After computing `amp_obs_flat` and before adding `extras["amp_obs"]`.

**What**

* Keep existing finite-value checks.
* Additionally:

  * Clamp `amp_obs_flat` to a reasonable range (e.g. `[-100, 100]`).
  * Apply `torch.nan_to_num` (or equivalent) on `amp_obs_flat`.
* Goal: prevent AMP-specific observations from injecting NaNs into the agent.

---

## 6. Normalization / RunningMeanStd hardening

**Where**

* `learning/common_agent.py`:

  * `_preproc_obs` (used to process observations before normalization).
* Optional: `rl_games.algos_torch.running_mean_std.RunningMeanStd.update`.

**What**

* In `_preproc_obs`:

  * Before passing observations into `self.obs_mean_std` (RunningMeanStd), apply:

    * `torch.nan_to_num`
    * Optional clamping (e.g. `[-100, 100]`).
* Optionally, in `RunningMeanStd.update`:

  * Skip or zero out any non-finite entries before computing batch mean/variance.
* Goal: ensure the global normalizer is never updated using NaNs/Inf from a single bad environment.

---

## 7. PPO / rl_games regularisation parameters

**Where**

* Training config YAML under `params.config` (used by rl_games PPO).

**What**

* Ensure the following settings (or equivalent) are present:

  * `normalize_input: true`
  * `grad_clip: 1.0` (or similar)
  * `bound_loss_type: "regularisation"`
  * `bounds_loss_coef: 0.001` (or similar small value)
* Goal: reduce policy explosions when occasional spikes survive earlier defences.

---

## 8. Conditional AMP discriminator (C-ASE)

**Where**

1. **Env side**

   * `env/tasks/humanoid_amp.py`:

     * `_setup_character_props` (defines `_num_amp_obs_per_step`).
     * `_compute_amp_observations`.
     * `_init_amp_obs_ref`.
     * `build_amp_obs_demo` and any functions filling `_amp_obs_buf`.

2. **Agent side**

   * `learning/amp_agent.py`:

     * `_build_net_config` (sets `config['amp_input_shape']`).
     * `_preproc_amp_obs`.
     * `_eval_disc`.

3. **Network side**

   * `learning/amp_network_builder.py`.
   * `learning/amp_models.py` (discriminator input shape).

**What**

* Choose a **shape vector** (e.g. SMPL betas) per environment:

  * Use existing per-env betas (e.g. `self._betas_env` or `template_betas[template_ids_env]`).
* For every AMP observation (both agent and demo/reference):

  * Concatenate the shape vector to the existing AMP observation at the last dimension.
* Increase `_num_amp_obs_per_step` accordingly and propagate the new dimension:

  * `get_num_amp_obs()` must match the augmented AMP observation size.
  * Ensure `amp_input_shape` in `_build_net_config` reflects this new size, so networks adjust automatically.
* Goal: discriminator becomes (D(s, s', \theta)), conditioned on morphology, avoiding manifold mismatch.

---

## 9. Shape-consistent retargeting of reference motions

**Where**

1. **Humanoid AMP task (passing shapes into motion library)**

   * `env/tasks/humanoid_amp.py`:

     * Where `MotionLibSMPL` is constructed and `load_motions` is called.
     * Currently uses a neutral `gender_beta` / zero betas for all envs.

2. **Motion library (kinematic retargeting logic)**

   * `utils/motion_lib_smpl.py` (or equivalent):

     * Methods that load mocap and implement `get_motion_state(motion_ids, motion_times, ...)`.

**What**

* In `HumanoidAMP`:

  * Replace the neutral/zero betas with the **true per-env SMPL betas**:

    * Build a tensor `[num_envs, beta_dim]` from `self._betas_env` (or template betas via `template_ids_env`).
    * Pad if necessary to match the beta dimension expected by `MotionLibSMPL`.
  * Pass this per-env shape tensor as `gender_betas` (or equivalent) into `load_motions`.

* In `MotionLibSMPL`:

  * When computing reference poses in `get_motion_state` (or similar):

    * Use the shape parameters for the corresponding environment to:

      * Run SMPL forward kinematics with that env’s betas, **or**
      * Apply a consistent bone-length scaling scheme (PHC-style) per shape.
    * Ensure root height and key positions are recomputed after scaling.

* Goal: reference (“Real”) motions are **retargeted/kinematically scaled** to each body shape, so they live on the same morphology manifold as the agent’s (“Fake”) motions.

---

## 10. AMP replay and gradient-side NaN defences

**Where**

* `learning/amp_agent.py`:

  * `play_steps`.
  * `calc_gradients`.
  * `_preproc_amp_obs`.
  * `_store_replay_amp_obs`.

**What**

* Keep existing NaN/Inf guards on:

  * `obs`, `amp_obs`, `advantages`, `returns`, `actions`.
* Extend checks where needed:

  * When reading `infos["amp_obs"]` or other extras, sanitize them before storing in the replay buffer.
* Goal: ensure that even if a rare NaN passes through the environment, it is filtered before influencing gradients.

---

## 11. Template-level debugging and pruning

**Where**

* `env/tasks/humanoid.py` / `env/tasks/humanoid_amp.py`:

  * Use of `_template_betas`, `_template_ids_env`, and logging around problematic env IDs or templates.

**What**

* Keep (or add) logging to report:

  * Template index, SMPL file name, and env ID whenever non-finite physics is detected.
* If specific templates systematically cause failures:

  * Either fix their XML/mesh (inertia, collisions), or temporarily remove them from the training set.
* Goal: isolate and remove pathological templates that systematically destabilize training.
