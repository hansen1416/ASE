The approach you described is feasible and aligns with common practices in Isaac Gym for domain randomization (e.g., varying physical parameters like body shapes across parallel environments). Since the current setup uses a fixed SMPL humanoid asset (from `smpl_humanoid.xml` in `humanoid_ase_smpl.yaml`), which likely assumes a neutral beta (all zeros, meaning average body shape), you'll need to extend it for variable betas. This involves:

- Generating multiple humanoid assets with different betas (using a tool like SMPLSim).
- Loading them as multiple asset handles in the environment.
- Assigning random assets (thus random betas) to environments during creation (or reset, though fixed-per-env is more efficient).
- Including betas in the task observations (and potentially AMP observations for the discriminator).
- Minor tweaks to the network input shapes and configs.

This will allow the policy to condition on betas (learning shape-dependent behaviors) and expose the model to diverse body shapes during training. Note that SMPL betas (typically 10 dimensions) affect mesh geometry and joint placements slightly, but the degrees of freedom (DoFs) remain the same (e.g., 66 for SMPL humanoid), so the action space doesn't change.

Below, I'll outline the step-by-step modifications, focusing on the `HumanoidAMP` task (likely in `ase/rlgpu/tasks/humanoid_amp.py` or a similar file in the ASE repo), environment config, and RL code. These are based on standard Isaac Gym task structure and the provided code snippets (e.g., `amp_agent.py`, `amp_network_builder.py`). If you have the full repo, verify the exact file paths.

### Step 1: Generate Multiple Humanoid Assets with Variable Betas
- The current asset (`mjcf/smpl_humanoid.xml`) is static. To vary betas, pre-generate multiple MJCF XML files with different betas sampled from a distribution (e.g., Gaussian around zero, as in SMPL datasets like AMASS).
- Use the SMPLSim library (pip-installable from https://github.com/ZhengyiLuo/SMPLSim), which supports creating Isaac Gym-compatible SMPL humanoids with custom betas.

Example script to generate assets (adapt from SMPLSim docs):
```python
import smpl_sim as ss
import numpy as np
import os

asset_root = "ase/data/assets/mjcf"  # Your asset directory
num_variants = 100  # e.g., 100 different body shapes
beta_std = 1.0  # Standard deviation for sampling (adjust based on realism)

for i in range(num_variants):
    betas = np.random.normal(0, beta_std, size=10)  # SMPL typically uses 10 betas
    humanoid = ss.Humanoid(  # Or HumanoidSMPL if specific
        betas=betas,
        gender='neutral',  # Or 'male'/'female'
        height=None,  # Auto from betas
    )
    xml_path = os.path.join(asset_root, f"smpl_humanoid_beta_{i}.xml")
    humanoid.export(xml_path, simulator='isaacgym')  # Exports MJCF for Isaac Gym
```

- This creates files like `smpl_humanoid_beta_0.xml`, `smpl_humanoid_beta_1.xml`, etc.
- Store the corresponding betas array for each file (e.g., in a dict or .npy file) so you can retrieve them later for observations.

### Step 2: Modify the Environment Config (`humanoid_ase_smpl.yaml`)
- Instead of a single `assetFileName`, add a list of asset files for randomization.
- Add beta-related params if needed.

Updated YAML snippet:
```yaml
env:
  # ... (keep existing)
  asset:
    assetRoot: "ase/data/assets"
    assetFiles:  # New: list of variants
      - "mjcf/smpl_humanoid_beta_0.xml"
      - "mjcf/smpl_humanoid_beta_1.xml"
      # ... add all 100 or sample a subset
  betaDim: 10  # New: dimension of betas (for obs space)
  randomizeBetas: True  # New: flag to enable randomization
```

- This is custom—you'll parse it in the task code below.

### Step 3: Modify the HumanoidAMP Task Class
Assuming the class is `HumanoidAMP` in `ase/rlgpu/tasks/humanoid_amp.py` (based on ASE repo structure). Key changes:

- **Asset Loading**: Load multiple assets in `__init__` or `_load_task_assets`.
- **Beta Storage**: Create a per-env tensor for betas.
- **Observation Construction**: Concat betas to task obs (and optionally AMP obs).
- **Reset Logic**: If randomizing per reset, re-assign actors (expensive); prefer fixed per env.

Example modified code (pseudocode based on standard Isaac Gym tasks like `Humanoid` in `isaacgymenvs`):
```python
from isaacgym import gymapi, gymtorch
import torch
import numpy as np
from .humanoid import Humanoid  # Base class

class HumanoidAMP(Humanoid):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        self._beta_dim = self.cfg.env.get('betaDim', 0)
        self._randomize_betas = self.cfg.env.get('randomizeBetas', False)
        self._asset_files = self.cfg.asset.get('assetFiles', [self.cfg.asset.assetFileName])
        self._num_asset_variants = len(self._asset_files)
        
        # Load all asset variants
        self._asset_handles = []
        self._betas_list = []  # Load from your .npy or dict
        for file in self._asset_files:
            asset = self.gym.load_asset(self.sim, self.asset_root, file, self.asset_options)
            self._asset_handles.append(asset)
            betas = np.load(f"ase/data/assets/mjcf/{file.replace('.xml', '_betas.npy')}")  # Your saved betas
            self._betas_list.append(betas)
        
        # Will be filled in create_sim
        self._per_env_asset_idx = None
        self.betas = None  # torch.tensor (num_envs, beta_dim)

    def create_sim(self):
        super().create_sim()  # Or override fully
        # Assign random asset to each env
        self._per_env_asset_idx = torch.randint(0, self._num_asset_variants, (self.num_envs,), device=self.device)
        self.betas = torch.zeros((self.num_envs, self._beta_dim), device=self.device)
        for i in range(self.num_envs):
            asset_idx = self._per_env_asset_idx[i]
            # Create actor with random asset
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 1.0, 0.0)  # Example start pose
            self.gym.create_actor(self.envs[i], self._asset_handles[asset_idx], pose, "humanoid", i, 0)
            self.betas[i] = torch.tensor(self._betas_list[asset_idx], device=self.device)

    def get_observation_space(self):
        obs_space = super().get_observation_space()
        # Increase shape for betas
        low, high = obs_space.low, obs_space.high
        new_shape = list(obs_space.shape)
        new_shape[-1] += self._beta_dim  # Concat to last dim
        return gym.spaces.Box(low=np.concatenate([low, -np.inf * np.ones(self._beta_dim)]), 
                              high=np.concatenate([high, np.inf * np.ones(self._beta_dim)]))

    def get_amp_observation_space(self):
        amp_space = super().get_amp_observation_space()  # e.g., (numAMPObsSteps * num_dof,)
        if self._condition_disc_on_beta:  # Optional flag you can add
            new_shape = list(amp_space.shape)
            new_shape[-1] += self._beta_dim * self.cfg.env.numAMPObsSteps  # Repeat betas per frame?
            return gym.spaces.Box(amp_space.low - np.inf, amp_space.high + np.inf, shape=new_shape)
        return amp_space

    def get_obs(self, env_ids=None):
        obs = super().get_obs(env_ids)  # Standard: root, joints, vels, contacts, etc.
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        # Concat betas
        obs = torch.cat([obs, self.betas[env_ids]], dim=-1)
        return obs

    def get_amp_obs(self, env_ids=None):
        amp_obs = super().get_amp_obs(env_ids)  # History of joint pos/vel (num_steps * dof)
        if self._condition_disc_on_beta:  # Optional: condition discriminator on shape
            repeated_betas = self.betas[env_ids].unsqueeze(1).repeat(1, self.cfg.env.numAMPObsSteps, 1)
            repeated_betas = repeated_betas.view(len(env_ids), -1)  # Flatten
            amp_obs = torch.cat([amp_obs, repeated_betas], dim=-1)
        return amp_obs

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if self._randomize_betas:  # Optional: resample per reset (slow!)
            for i in env_ids:
                # Destroy old actor
                actor_handle = self.gym.get_actor_handle(self.envs[i], 0)  # Assuming one actor
                self.gym.destroy_actor(self.envs[i], actor_handle)
                # Re-create with new random asset
                new_asset_idx = torch.randint(0, self._num_asset_variants, (1,), device=self.device)[0]
                pose = gymapi.Transform()  # Reset pose
                self.gym.create_actor(self.envs[i], self._asset_handles[new_asset_idx], pose, "humanoid", i, 0)
                self._per_env_asset_idx[i] = new_asset_idx
                self.betas[i] = torch.tensor(self._betas_list[new_asset_idx], device=self.device)
```

- **Notes**:
  - Fixed-per-env assignment (during `create_sim`) is efficient for 4096 envs—ensures diversity across batch.
  - Per-reset randomization: Possible but GPU-expensive (actor creation/destruction); avoid if possible.
  - Conditioning discriminator on betas: Optional but recommended if shapes affect "natural" motions. Repeat betas across the AMP history frames.

### Step 4: Update RL Code and Configs
- **Observations**: The env now has larger `obs_shape` and `amp_observation_space` (includes betas). This propagates to `amp_agent.py` (`self.obs_shape`, `self._amp_observation_space`).
- **Network**: In `amp_network_builder.py`, input_shape includes +10 for betas. No change needed—it uses config['input_shape'] from env.
- **AMP Normalization**: In `amp_agent.py` / `amp_players.py`, `_normalize_amp_input` handles the larger shape automatically.
- **Config Updates** (`amp_humanoid.yaml`):
  ```yaml
  config:
    # ... (keep existing)
    normalize_input: True  # Already true; handles larger obs
    amp_input_shape: # Optional: override if needed, but env sets it
  ```
- **Training**: No major change—the agent sees betas in obs, learns to condition policy. Disc rewards adapt to shape-dependent motions.

### Potential Issues and Tips
- **Performance**: Generating/loading 100 assets may be memory-heavy; start with 10-20 variants.
- **Retargeting Motions**: If using demo motions (e.g., `motion_file`), retarget them to different shapes (use ASE's `poselib/retarget_motion.py` with betas).
- **C·ASE Extension**: If you want advanced conditioning (like in C·ASE paper), embed betas similarly (e.g., MLP to 64D, concat like skill labels c). But for basic, direct concat works.
- **Testing**: Run with `--num_envs 16 --headless False` to visualize different shapes.
- **Repo Alignment**: Confirm `HumanoidAMP` code in ASE repo—it's similar to NVIDIA's `isaacgymenvs/tasks/humanoid_amp.py`. If stuck, fork and test.

This should get you a model conditioned on betas with diverse training shapes. If you share more code (e.g., the exact HumanoidAMP file), I can refine.