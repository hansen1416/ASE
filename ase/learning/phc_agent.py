import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import numpy as np
from rl_games.common import a2c_common

import learning.amp_agent as AMPAgent

@dataclass
class MultiMotionCfg:
    # A) visibility + control
    track_motion_stats: bool = True
    motion_stats_topk: int = 20
    store_motion_ids_in_rollout: bool = False  # store per-step motion id tensor (optional)

    eval_enabled: bool = True
    eval_freq: int = 10                # evaluate every N epochs (0 disables auto-eval)
    eval_num_episodes: int = 0          # 0 => default to num_envs
    eval_deterministic: bool = True
    eval_max_steps: int = 0            # 0 => no hard cap

    # B) stability
    norm_disc_reward: bool = False
    disc_reward_norm_scale: float = 1.0    # optional affine after RMS
    disc_reward_norm_shift: float = 0.0

    freeze_obs_rms_per_epoch: bool = False


class MotionStats:
    """
    Per-motion episode statistics.
    Uses CPU tensors; updates are cheap because done batch is small.
    """
    def __init__(self, num_motions: int):
        self.num_motions = int(num_motions)
        self.episodes = torch.zeros(num_motions, dtype=torch.long)
        self.terminations = torch.zeros(num_motions, dtype=torch.long)
        self.successes = torch.zeros(num_motions, dtype=torch.long)
        self.return_sum = torch.zeros(num_motions, dtype=torch.float32)
        self.len_sum = torch.zeros(num_motions, dtype=torch.float32)

    def update(
        self,
        motion_ids: torch.Tensor,        # (K,) CPU long
        terminated: torch.Tensor,        # (K,) CPU bool
        ep_returns: torch.Tensor,        # (K,) CPU float
        ep_lens: torch.Tensor,           # (K,) CPU float
    ):
        motion_ids = motion_ids.long()
        ones = torch.ones_like(motion_ids, dtype=torch.long)

        self.episodes.index_add_(0, motion_ids, ones)
        term_long = terminated.long()
        self.terminations.index_add_(0, motion_ids, term_long)
        self.successes.index_add_(0, motion_ids, (1 - term_long))

        self.return_sum.index_add_(0, motion_ids, ep_returns.float())
        self.len_sum.index_add_(0, motion_ids, ep_lens.float())

    def summary(self, topk: int = 20) -> Dict[str, Any]:
        seen = self.episodes > 0
        coverage = int(seen.sum().item())
        total_eps = int(self.episodes.sum().item())
        if total_eps == 0:
            return {
                "coverage": 0,
                "total_episodes": 0,
                "termination_rate": 0.0,
                "top_fail_ids": [],
                "top_fail_rates": [],
            }

        term_rate = (self.terminations.sum().float() / self.episodes.sum().float()).item()

        # per-motion termination rate, ignore unseen
        rates = torch.full_like(self.return_sum, -1.0)
        rates[seen] = self.terminations[seen].float() / self.episodes[seen].float()

        k = min(int(topk), coverage)
        vals, idx = torch.topk(rates, k=k)
        return {
            "coverage": coverage,
            "total_episodes": total_eps,
            "termination_rate": float(term_rate),
            "top_fail_ids": idx.cpu().tolist(),
            "top_fail_rates": vals.cpu().tolist(),
        }


class AMPAgentMultiMotion(AMPAgent):
    """
    Drop-in AMPAgent variant for many motions:
      A) motion stats + eval
      B) disc reward RMS + frozen obs RMS per epoch
      C) AMP-aware play_steps_rnn
    """

    # ---- config ----
    def _load_config_params(self, config):
        super()._load_config_params(config)

        mm = config.get("multi_motion", {}) or {}
        self._mm_cfg = MultiMotionCfg(
            track_motion_stats=bool(mm.get("track_motion_stats", True)),
            motion_stats_topk=int(mm.get("motion_stats_topk", 20)),
            store_motion_ids_in_rollout=bool(mm.get("store_motion_ids_in_rollout", False)),

            eval_enabled=bool(mm.get("eval_enabled", True)),
            eval_freq=int(mm.get("eval_freq", 10)),
            eval_num_episodes=int(mm.get("eval_num_episodes", 0)),
            eval_deterministic=bool(mm.get("eval_deterministic", True)),
            eval_max_steps=int(mm.get("eval_max_steps", 0)),

            norm_disc_reward=bool(mm.get("norm_disc_reward", False)),
            disc_reward_norm_scale=float(mm.get("disc_reward_norm_scale", 1.0)),
            disc_reward_norm_shift=float(mm.get("disc_reward_norm_shift", 0.0)),

            freeze_obs_rms_per_epoch=bool(mm.get("freeze_obs_rms_per_epoch", False)),
        )
        return

    def __init__(self, base_name, config):
        super().__init__(base_name, config)

        # B1) optional discriminator reward RMS
        if self._mm_cfg.norm_disc_reward:
            self._disc_reward_mean_std = RunningMeanStd((1,)).to(self.ppo_device)
        else:
            self._disc_reward_mean_std = None

        # B2) frozen obs RMS snapshot
        self._obs_rms_snapshot = None
        self._use_obs_rms_snapshot = False

        # A2) stats tracker (init later, when motion_lib is available)
        self._motion_stats: Optional[MotionStats] = None

    # ---- weights / modes ----
    def set_eval(self):
        super().set_eval()
        if self._disc_reward_mean_std is not None:
            self._disc_reward_mean_std.eval()

    def set_train(self):
        super().set_train()
        if self._disc_reward_mean_std is not None:
            self._disc_reward_mean_std.train()

    def get_stats_weights(self):
        state = super().get_stats_weights()
        if self._disc_reward_mean_std is not None:
            state["disc_reward_mean_std"] = self._disc_reward_mean_std.state_dict()
        return state

    def set_stats_weights(self, weights):
        super().set_stats_weights(weights)
        if self._disc_reward_mean_std is not None and "disc_reward_mean_std" in weights:
            self._disc_reward_mean_std.load_state_dict(weights["disc_reward_mean_std"])

    # ---- A1) motion id access ----
    def _get_motion_ids(self, infos: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        if infos is not None and "motion_ids" in infos:
            mids = infos["motion_ids"]
            if torch.is_tensor(mids):
                return mids.to(self.device)
            return torch.as_tensor(mids, device=self.device, dtype=torch.long)

        task = self.vec_env.env.task
        if hasattr(task, "_sampled_motion_ids"):
            mids = task._sampled_motion_ids
            return mids.to(self.device) if torch.is_tensor(mids) else torch.as_tensor(mids, device=self.device, dtype=torch.long)

        raise RuntimeError("Cannot obtain motion_ids: provide infos['motion_ids'] or task._sampled_motion_ids.")

    def _try_init_motion_stats(self):
        if not self._mm_cfg.track_motion_stats:
            return
        if self._motion_stats is not None:
            return

        task = self.vec_env.env.task
        motion_lib = getattr(task, "_motion_lib", None)
        if motion_lib is None:
            return

        # best-effort: common naming patterns
        num_motions = None
        for attr in ["num_motions", "get_num_motions", "_num_motions"]:
            if hasattr(motion_lib, attr):
                v = getattr(motion_lib, attr)
                num_motions = int(v() if callable(v) else v)
                break

        if num_motions is None:
            # fallback: do not crash; disable per-motion stats
            print("[AMPAgentMultiMotion] motion_lib has no num_motions; disabling per-motion stats.")
            self._mm_cfg.track_motion_stats = False
            return

        self._motion_stats = MotionStats(num_motions)

    # ---- B2) obs RMS snapshot ----
    def _begin_epoch_obs_rms_snapshot(self):
        if not self._mm_cfg.freeze_obs_rms_per_epoch:
            return
        if not hasattr(self, "running_mean_std") or self.running_mean_std is None:
            return

        self._obs_rms_snapshot = copy.deepcopy(self.running_mean_std)
        # ensure snapshot does not update
        if hasattr(self._obs_rms_snapshot, "freeze"):
            self._obs_rms_snapshot.freeze()
        else:
            self._obs_rms_snapshot.eval()

    def _end_epoch_obs_rms_snapshot(self):
        self._obs_rms_snapshot = None
        self._use_obs_rms_snapshot = False

    # override obs preprocessing to optionally use the snapshot without copying base logic
    def _preproc_obs(self, obs_batch):
        if not self._use_obs_rms_snapshot or self._obs_rms_snapshot is None:
            return super()._preproc_obs(obs_batch)

        # temporarily swap running_mean_std -> snapshot, call base, then restore
        old = self.running_mean_std
        self.running_mean_std = self._obs_rms_snapshot
        try:
            return super()._preproc_obs(obs_batch)
        finally:
            self.running_mean_std = old

    # ---- B1) disc reward normalization ----
    def _norm_disc_reward(self) -> bool:
        return self._disc_reward_mean_std is not None

    def _calc_disc_rewards(self, amp_obs):
        with torch.no_grad():
            disc_logits = self._eval_disc(amp_obs)
            prob = 1 / (1 + torch.exp(-disc_logits))
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.ppo_device)))

            if self._norm_disc_reward():
                # update RMS during training; during eval it will be in eval() mode
                self._disc_reward_mean_std.train()
                flat = disc_r.flatten()
                norm = self._disc_reward_mean_std(flat)
                disc_r = norm.reshape(disc_r.shape)
                disc_r = self._mm_cfg.disc_reward_norm_scale * disc_r + self._mm_cfg.disc_reward_norm_shift

            disc_r *= self._disc_reward_scale

        return disc_r

    # ---- A3) deterministic get_action + eval step ----
    def get_action(self, obs_dict: Dict[str, torch.Tensor], deterministic: bool = True) -> torch.Tensor:
        processed_obs = self._preproc_obs(obs_dict["obs"])
        self.model.eval()
        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": processed_obs,
            "rnn_states": self.rnn_states,
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)

        if "rnn_states" in res_dict:
            self.rnn_states = res_dict["rnn_states"]

        return res_dict["mus"] if deterministic else res_dict["actions"]

    def env_eval_step(self, actions: torch.Tensor):
        # keep consistent with your env_step handling
        return self.env_step(actions)

    def eval(self) -> Dict[str, float]:
        if not self._mm_cfg.eval_enabled:
            return {}

        self.set_eval()
        self._try_init_motion_stats()

        num_envs = self.num_actors
        num_eps = self._mm_cfg.eval_num_episodes if self._mm_cfg.eval_num_episodes > 0 else num_envs

        ep_ret = torch.zeros(num_envs, device=self.device)
        ep_len = torch.zeros(num_envs, device=self.device)

        done_indices = []
        finished = 0
        term_list = []
        mid_list = []

        steps = 0
        while finished < num_eps:
            if self._mm_cfg.eval_max_steps > 0 and steps >= self._mm_cfg.eval_max_steps:
                break
            steps += 1

            obs_dict = self.env_reset(done_indices)
            act = self.get_action(obs_dict, deterministic=self._mm_cfg.eval_deterministic)
            obs_dict, rewards, dones, infos = self.env_eval_step(act)

            ep_ret += rewards
            ep_len += 1

            terminated = infos.get("terminate", torch.zeros_like(dones)).bool()
            all_done = dones.nonzero(as_tuple=False)
            done_envs = all_done[:: self.num_agents][:, 0] if len(all_done) > 0 else None

            if done_envs is None or done_envs.numel() == 0:
                continue

            mids = self._get_motion_ids(infos)
            term_done = terminated[done_envs].detach().cpu()
            mid_done = mids[done_envs].detach().cpu()

            term_list.append(term_done)
            mid_list.append(mid_done)

            # reset accumulators for those envs
            ep_ret[done_envs] = 0
            ep_len[done_envs] = 0

            # reset RNN state on done
            if self.is_rnn and self.rnn_states is not None:
                for s in self.rnn_states:
                    s[:, all_done, :] = 0.0

            finished += int(done_envs.numel())
            done_indices = done_envs

        if len(term_list) == 0:
            return {
                "eval/success_rate": 0.0,
                "eval/termination_rate": 0.0,
                "eval/coverage": 0.0,
            }

        term_all = torch.cat(term_list, dim=0).float()
        mids_all = torch.cat(mid_list, dim=0)

        term_rate = term_all.mean().item()
        succ_rate = (1.0 - term_rate)

        coverage = float(len(torch.unique(mids_all))) if mids_all.numel() > 0 else 0.0

        return {
            "eval/success_rate": float(succ_rate),
            "eval/termination_rate": float(term_rate),
            "eval/coverage": coverage,
        }

    # ---- A2) update motion stats on episode completion ----
    def _update_motion_stats_on_done(self, done_envs: torch.Tensor, infos: Dict[str, Any], terminated: torch.Tensor):
        if not self._mm_cfg.track_motion_stats:
            return
        self._try_init_motion_stats()
        if self._motion_stats is None:
            return

        mids = self._get_motion_ids(infos)[done_envs].detach().cpu().long()
        term = terminated[done_envs].detach().cpu().bool()

        # episode returns/lengths from agent accumulators
        # current_rewards is (num_env, 1) in rl_games agents; be robust
        ep_r = self.current_rewards[done_envs].detach().cpu()
        if ep_r.ndim > 1:
            ep_r = ep_r.squeeze(-1)
        ep_l = self.current_lengths[done_envs].detach().cpu().float()

        self._motion_stats.update(mids, term, ep_r, ep_l)

    # ---- C) AMP-aware play_steps_rnn ----
    def play_steps_rnn(self):
        """
        RNN rollout with AMP buffers + terminate-masked next_values + rand_action_mask.
        Mirrors your non-RNN play_steps() semantics.
        """
        self.set_eval()

        mb_rnn_states = []
        self.experience_buffer.tensor_dict["values"].fill_(0)
        self.experience_buffer.tensor_dict["rewards"].fill_(0)
        self.experience_buffer.tensor_dict["dones"].fill_(1)

        update_list = self.update_list
        batch_size = self.num_agents * self.num_actors

        mb_rnn_masks = None
        mb_rnn_masks, indices, steps_mask, steps_state, play_mask, mb_rnn_states = self.init_rnn_step(
            batch_size, mb_rnn_states
        )

        done_indices = []

        for n in range(self.horizon_length):
            self.obs = self.env_reset(done_indices)

            seq_indices, full_tensor = self.process_rnn_indices(
                mb_rnn_masks, indices, steps_mask, steps_state, mb_rnn_states
            )
            if full_tensor:
                break

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs, self._rand_action_probs)

            self.rnn_states = res_dict["rnn_states"]
            self.experience_buffer.update_data_rnn("obses", indices, play_mask, self.obs["obs"])

            for k in update_list:
                self.experience_buffer.update_data_rnn(k, indices, play_mask, res_dict[k])

            if self.has_central_value:
                self.experience_buffer.update_data_rnn(
                    "states",
                    indices[:: self.num_agents],
                    play_mask[:: self.num_agents] // self.num_agents,
                    self.obs["states"],
                )

            self.obs, rewards, self.dones, infos = self.env_step(res_dict["actions"])
            shaped_rewards = self.rewards_shaper(rewards)

            self.experience_buffer.update_data_rnn("rewards", indices, play_mask, shaped_rewards)
            self.experience_buffer.update_data_rnn("next_obses", indices, play_mask, self.obs["obs"])
            self.experience_buffer.update_data_rnn("dones", indices, play_mask, self.dones.byte())
            self.experience_buffer.update_data_rnn("amp_obs", indices, play_mask, infos["amp_obs"])
            self.experience_buffer.update_data_rnn("rand_action_mask", indices, play_mask, res_dict["rand_action_mask"])

            terminated = infos["terminate"].float()
            terminated = terminated.unsqueeze(-1)

            # critic eval with rnn_states
            input_dict = {"obs": self.obs["obs"], "rnn_states": self.rnn_states}
            next_vals = self._eval_critic(input_dict)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data_rnn("next_values", indices, play_mask, next_vals)

            # bookkeeping
            self.current_rewards += rewards
            self.current_lengths += 1

            all_done = self.dones.nonzero(as_tuple=False)
            done_envs = all_done[:: self.num_agents][:, 0] if len(all_done) > 0 else None

            # update per-motion stats
            if done_envs is not None and done_envs.numel() > 0:
                self._update_motion_stats_on_done(done_envs, infos, infos["terminate"].bool())

            self.process_rnn_dones(all_done, indices, seq_indices)

            self.algo_observer.process_infos(infos, all_done[:: self.num_agents])

            not_dones = 1.0 - self.dones.float()
            self.game_rewards.update(self.current_rewards[all_done[:: self.num_agents]])
            self.game_lengths.update(self.current_lengths[all_done[:: self.num_agents]])

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

            done_indices = done_envs if done_envs is not None else []

        mb_fdones = self.experience_buffer.tensor_dict["dones"].float()
        mb_values = self.experience_buffer.tensor_dict["values"]
        mb_next_values = self.experience_buffer.tensor_dict["next_values"]
        mb_rewards = self.experience_buffer.tensor_dict["rewards"]
        mb_amp_obs = self.experience_buffer.tensor_dict["amp_obs"]

        amp_rewards = self._calc_amp_rewards(mb_amp_obs)
        mb_rewards = self._combine_rewards(mb_rewards, amp_rewards)

        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict["returns"] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict["rnn_states"] = mb_rnn_states
        batch_dict["rnn_masks"] = mb_rnn_masks
        batch_dict["played_frames"] = n * self.num_actors * self.num_agents

        for k, v in amp_rewards.items():
            batch_dict[k] = a2c_common.swap_and_flatten01(v)

        batch_dict["mb_rewards"] = a2c_common.swap_and_flatten01(mb_rewards)
        return batch_dict

    # ---- override train_epoch to insert B2 snapshot + optional eval + stats summary ----
    def train_epoch(self):
        self._begin_epoch_obs_rms_snapshot()
        self._try_init_motion_stats()

        play_time_start = time.time()
        with torch.no_grad():
            self._use_obs_rms_snapshot = False  # rollout uses normal eval-time RMS
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()
        play_time_end = time.time()

        update_time_start = time.time()

        self._update_amp_demos()
        num_obs_samples = batch_dict["amp_obs"].shape[0]
        batch_dict["amp_obs_demo"] = self._amp_obs_demo_buffer.sample(num_obs_samples)["amp_obs"]
        batch_dict["amp_obs_replay"] = (
            batch_dict["amp_obs"]
            if (self._amp_replay_buffer.get_total_count() == 0)
            else self._amp_replay_buffer.sample(num_obs_samples)["amp_obs"]
        )

        self.set_train()
        self._use_obs_rms_snapshot = bool(self._mm_cfg.freeze_obs_rms_per_epoch)

        self.curr_frames = batch_dict.pop("played_frames")
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        if self.has_central_value:
            self.train_central_value()

        train_info = None
        for _ in range(0, self.mini_epochs_num):
            for i in range(len(self.dataset)):
                curr_train_info = self.train_actor_critic(self.dataset[i])

                if train_info is None:
                    train_info = {k: [v] for k, v in curr_train_info.items()}
                else:
                    for k, v in curr_train_info.items():
                        train_info[k].append(v)

        update_time_end = time.time()

        self._store_replay_amp_obs(batch_dict["amp_obs"])

        train_info["play_time"] = play_time_end - play_time_start
        train_info["update_time"] = update_time_end - update_time_start
        train_info["total_time"] = update_time_end - play_time_start
        train_info["mb_rewards"] = batch_dict.get("mb_rewards", None)

        # A2 summary
        if self._motion_stats is not None:
            ms = self._motion_stats.summary(topk=self._mm_cfg.motion_stats_topk)
            train_info["motion/coverage"] = ms["coverage"]
            train_info["motion/termination_rate"] = ms["termination_rate"]
            train_info["motion/top_fail_ids"] = ms["top_fail_ids"]
            train_info["motion/top_fail_rates"] = ms["top_fail_rates"]

        # A3 optional auto-eval
        if self._mm_cfg.eval_enabled and self._mm_cfg.eval_freq > 0:
            if (self.epoch_num % self._mm_cfg.eval_freq) == 0:
                eval_info = self.eval()
                train_info.update(eval_info)

        self._record_train_batch_info(batch_dict, train_info)

        self._end_epoch_obs_rms_snapshot()
        return train_info
