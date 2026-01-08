# phc_agent_min.py
import torch
from rl_games.common import a2c_common

import learning.amp_agent as amp_agent


class PHCAgent(amp_agent.AMPAgent):
    """
    Training-only extension of AMPAgent.

    Keeps ONLY what AMPAgent is missing for training:
      - play_steps_rnn(): AMP-aware RNN rollout that stores amp_obs and rand_action_mask,
        and masks critic bootstrapping with infos['terminate'].

    Everything else (eval loops, motion stats, extra RMS tricks) is intentionally removed.
    """

    def play_steps_rnn(self):
        """
        RNN rollout with AMP bookkeeping.

        Key differences vs a generic rl-games RNN rollout:
          1) Store infos["amp_obs"] into the experience buffer.
          2) Compute next_values with critic and apply terminate-masking:
                 next_vals *= (1 - terminate)
             This avoids bootstrapping through failure transitions (falls).
          3) Store rand_action_mask if you use eps-greedy action mixing.
        """
        self.set_eval()

        mb_rnn_states = []

        # Reset buffer contents to avoid leftover values from previous epoch.
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

            # Select the subset of envs that should step next (rl-games RNN scheduling).
            seq_indices, full_tensor = self.process_rnn_indices(
                mb_rnn_masks, indices, steps_mask, steps_state, mb_rnn_states
            )
            if full_tensor:
                break

            # Get action/value/logp/mu/sigma/... for the current observation.
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs, self._rand_action_probs)

            # Track RNN states for next timestep.
            self.rnn_states = res_dict["rnn_states"]

            # Store observation (obses) into the RNN-formatted experience buffer.
            self.experience_buffer.update_data_rnn("obses", indices, play_mask, self.obs["obs"])

            # Store standard PPO fields (actions, logp, values, mu, sigma, etc.).
            for k in update_list:
                self.experience_buffer.update_data_rnn(k, indices, play_mask, res_dict[k])

            if self.has_central_value:
                self.experience_buffer.update_data_rnn(
                    "states",
                    indices[:: self.num_agents],
                    play_mask[:: self.num_agents] // self.num_agents,
                    self.obs["states"],
                )

            # Step the environment with chosen actions.
            self.obs, rewards, self.dones, infos = self.env_step(res_dict["actions"])
            shaped_rewards = self.rewards_shaper(rewards)

            # Store transition fields.
            self.experience_buffer.update_data_rnn("rewards", indices, play_mask, shaped_rewards)
            self.experience_buffer.update_data_rnn("next_obses", indices, play_mask, self.obs["obs"])
            self.experience_buffer.update_data_rnn("dones", indices, play_mask, self.dones.byte())

            # AMP-specific: store amp observations used by discriminator reward and loss.
            self.experience_buffer.update_data_rnn("amp_obs", indices, play_mask, infos["amp_obs"])

            # Eps-greedy bookkeeping: which envs used random actions vs deterministic mu.
            self.experience_buffer.update_data_rnn("rand_action_mask", indices, play_mask, res_dict["rand_action_mask"])

            # Terminate flag indicates failure termination (e.g., fall) distinct from time-limit.
            terminated = infos["terminate"].float().unsqueeze(-1)

            # Critic bootstrap: evaluate V(s_{t+1}), but do NOT bootstrap through failure transitions.
            input_dict = {"obs": self.obs["obs"], "rnn_states": self.rnn_states}
            next_vals = self._eval_critic(input_dict)
            next_vals *= (1.0 - terminated)

            self.experience_buffer.update_data_rnn("next_values", indices, play_mask, next_vals)

            # Update per-env episode accumulators (used by logging and MotionStats).
            self.current_rewards += rewards
            self.current_lengths += 1

            all_done = self.dones.nonzero(as_tuple=False)
            done_envs = all_done[:: self.num_agents][:, 0] if len(all_done) > 0 else None

            # Update per-motion episode statistics before current_rewards/current_lengths get reset.
            if done_envs is not None and done_envs.numel() > 0:
                self._update_motion_stats_on_done(done_envs, infos, infos["terminate"].bool())

            # Handle RNN done logic and observer hooks.
            self.process_rnn_dones(all_done, indices, seq_indices)
            self.algo_observer.process_infos(infos, all_done[:: self.num_agents])

            # Update episode reward/length meters used by base logging.
            not_dones = 1.0 - self.dones.float()
            self.game_rewards.update(self.current_rewards[all_done[:: self.num_agents]])
            self.game_lengths.update(self.current_lengths[all_done[:: self.num_agents]])

            # Reset accumulators for finished envs (mask style).
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

            done_indices = done_envs if done_envs is not None else []

        # After rollout, compute AMP rewards and PPO advantages/returns.
        mb_fdones = self.experience_buffer.tensor_dict["dones"].float()
        mb_values = self.experience_buffer.tensor_dict["values"]
        mb_next_values = self.experience_buffer.tensor_dict["next_values"]
        mb_rewards = self.experience_buffer.tensor_dict["rewards"]
        mb_amp_obs = self.experience_buffer.tensor_dict["amp_obs"]

        amp_rewards = self._calc_amp_rewards(mb_amp_obs)
        mb_rewards = self._combine_rewards(mb_rewards, amp_rewards)

        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        # Flatten [T, N, ...] -> [T*N, ...] for training.
        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict["returns"] = a2c_common.swap_and_flatten01(mb_returns)

        # RNN training additionally needs stored rnn_states and masks.
        batch_dict["rnn_states"] = mb_rnn_states
        batch_dict["rnn_masks"] = mb_rnn_masks

        # Played frames is used by some schedulers/loggers.
        batch_dict["played_frames"] = n * self.num_actors * self.num_agents

        for k, v in amp_rewards.items():
            batch_dict[k] = a2c_common.swap_and_flatten01(v)

        batch_dict["mb_rewards"] = a2c_common.swap_and_flatten01(mb_rewards)
        return batch_dict
