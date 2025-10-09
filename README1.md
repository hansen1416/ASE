Yes—you can make ASE shape-aware. Minimal, safe changes:

What to change

Sample β per episode (or fix it)

In your humanoid task reset, draw β ~ p(β) (or load your custom β).

Build the sim robot with that β (modify _create_smpl_humanoid_xml or bypass it with a prebuilt MJCF for your β).

Regenerate the reference for the SAME β

For each rollout, compute reference joint positions/vels via SMPL(θ(t), β).

Do it on the fly (FK per step) or cache per episode.

Never mix sim-β and ref-β.

Expose β to the policy (and value)

Append β to proprioception: s_t := concat(q_t, q̇_t, β).

Update obs dim and normalization stats.

Expose β to the ASE discriminator

ASE’s discriminator judges short motion windows. Concatenate β to each frame’s disc input (or to the window embedding).

This prevents the disc from penalizing geometric differences that are purely due to limb lengths.

Conditioning stays in z; β is context

Keep ASE’s skill latent z as the “which motion” code.

Treat β as environment context (part of observation), not as a skill id.

Reward terms—keep geometry consistent

Position/orientation/velocity terms compare sim vs ref built with the same β.

If you ever must use ref from a different β, down-weight absolute position terms and rely more on orientations + AMP/ASE prior (not recommended if you can match β).

### Concretely (where to touch)

Env build: phc/env/tasks/humanoid.py

At reset, sample/set self.curr_betas and pass to robot build (patch _create_smpl_humanoid_xml to accept β; or load a custom MJCF produced for that β).

Reference provider: wherever ref states are produced (FK from θ)

Replace SMPL(θ, β=0) with SMPL(θ, β=self.curr_betas) for this episode.

Observation assembly: task’s compute_observations()

obs = torch.cat([q, qd, beta], dim=-1).

Discriminator input builder (ASE/AMP module)

When you gather motion windows for D, append β to each frame feature (or to the pooled window feature).

Config:

Add robot.shape.beta_mode: {fixed: [..], sample: {dist: “…”, range: …}}

Add flags like obs.include_beta: true, disc.include_beta: true.

### Performance tips

Cache per-episode FK: compute all p̂ₜ once at episode start (θ sequence is fixed), index by t.

Normalize β: zero-mean, unit-var over your sampling range to keep scales balanced.

Curriculum: start with a narrow β range, widen as the policy stabilizes.

### Summary

ASE will handle “all motions” more efficiently than training one AMP per clip. To support custom shapes, sample or fix β per episode, rebuild both the sim and the reference with that β, and feed β into both the policy and the discriminator. That’s it.