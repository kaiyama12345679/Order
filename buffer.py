from collections import namedtuple
import random
import torch
from valuenorm import ValueNorm

Transition = namedtuple(
    "Transition",
    (
        "obs",
        "values",
        "actions",
        "action_masks",
        "action_logprobs",
        "orders",
        "order_logprobs",
        "rewards",
        "dones",
        "active_masks",
        "advantages",
        "returns",
    ),
)


class EpisodeBuffer(object):
    def __init__(
        self,
        n_env,
        n_agents,
        obs_dim,
        action_dim,
        gamma,
        tau,
        episode_length,
        max_batch_size,
        device,
        shuffle_agent_idx,
    ):
        self.n_env = n_env
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Parameters
        self.gamma = gamma
        self.tau = tau
        self.episode_length = episode_length
        self.max_batch_size = max_batch_size
        self.counter = 0
        self.shuffle_agent_idx = shuffle_agent_idx

        # Required episode_length + 1
        self.value_preds = torch.zeros(
            (episode_length + 1, self.n_env, self.n_agents, 1), device=device
        )
        # Required episode_length
        self.obs = torch.zeros(
            (episode_length, self.n_env, self.n_agents, self.obs_dim), device=device
        )
        self.actions = torch.zeros(
            (episode_length, self.n_env, self.n_agents, 1), device=device
        )
        self.action_masks = torch.zeros(
            (episode_length, self.n_env, self.n_agents, self.action_dim), device=device
        )
        self.action_logprobs = torch.zeros(
            (episode_length, self.n_env, self.n_agents, 1), device=device
        )
        self.orders = torch.zeros(
            (episode_length, self.n_env, self.n_agents, 1), device=device
        )
        self.order_logprobs = torch.zeros(
            (episode_length, self.n_env, self.n_agents, 1), device=device
        )
        self.rewards = torch.zeros(
            (episode_length, self.n_env, self.n_agents, 1), device=device
        )
        self.dones = torch.zeros(
            (episode_length, self.n_env, self.n_agents, 1), device=device
        )
        self.active_masks = torch.zeros(
            (episode_length, self.n_env, self.n_agents, 1), device=device
        )

        # To be calculated from the above later
        self.advantages = torch.zeros(
            (episode_length, self.n_env,  self.n_agents, 1), device=device
        )
        self.returns = torch.zeros(
            (episode_length, self.n_env, self.n_agents, 1), device=device
        )

    def is_full(self):
        return self.counter == self.episode_length

    def insert(
        self,
        obs,
        action,
        action_mask,
        action_logprob,
        order,
        order_logprob,
        reward,
        done,
        value_preds,
    ):
        if self.is_full():
            raise RuntimeError("Buffer is full")
        self.obs[self.counter] = obs
        self.actions[self.counter] = action
        self.action_masks[self.counter] = action_mask
        self.action_logprobs[self.counter] = action_logprob
        self.orders[self.counter] = order
        self.order_logprobs[self.counter] = order_logprob
        self.rewards[self.counter] = reward

        # Done has information for each agent. Alive: 0.0, Die: 1.0
        # So 1.0 - Done will give you active masks
        # TODO: Check if active mask should be False if truncated == True?
        self.active_masks[self.counter] = 1.0 - done.unsqueeze(-1)
        # Here truncated is used as done flag
        self.dones[self.counter] = done.unsqueeze(-1)
        self.value_preds[self.counter] = value_preds

        # Counter add
        self.counter += 1

    def add_next_value(self, values):
        if not self.is_full():
            raise RuntimeError("Buffer is not full. So not ready for the last value")
        self.value_preds[-1] = values

    def compute_advantages(self, value_normalizer: ValueNorm = None):
        # must execute after an episode
        if not self.is_full():
            raise RuntimeError("Advantages should be computed when the buffer is full")
        gae = 0
        for t in reversed(range(self.episode_length)):
            if value_normalizer:
                next_value = value_normalizer.denormalize(self.value_preds[t + 1]) * (
                    1 - self.dones[t]
                )
                delta = (
                    self.rewards[t]
                    + self.gamma * next_value
                    - value_normalizer.denormalize(self.value_preds[t])
                )
            else:
                next_value = self.value_preds[t + 1] * (1 - self.dones[t])
                delta = self.rewards[t] + self.gamma * next_value - self.value_preds[t]
            gae = delta + self.gamma * self.tau * gae * (1 - self.dones[t])
            self.advantages[t] = gae
            if value_normalizer:
                self.returns[t] = gae + value_normalizer.denormalize(
                    self.value_preds[t]
                )
            else:
                self.returns[t] = gae + self.value_preds[t]

    def sample(self, isall=False):
        # Reshape
        # (n_episode, n_env, n_agent, dim) -> (n_episode * n_env, n_agent, dim)
        value_preds = self.value_preds[:-1].reshape(
            self.episode_length * self.n_env, self.n_agents, -1
        )
        obs = self.obs.reshape(self.episode_length * self.n_env, self.n_agents, -1)
        actions = self.actions.reshape(
            self.episode_length * self.n_env, self.n_agents, -1
        )
        action_masks = self.action_masks.reshape(
            self.episode_length * self.n_env, self.n_agents, -1
        )
        action_logprobs = self.action_logprobs.reshape(
            self.episode_length * self.n_env, self.n_agents, -1
        )
        orders = self.orders.reshape(
            self.episode_length * self.n_env, self.n_agents, -1
        )
        order_logprobs = self.order_logprobs.reshape(
            self.episode_length * self.n_env, self.n_agents, -1
        )
        rewards = self.rewards.reshape(
            self.episode_length * self.n_env, self.n_agents, -1
        )
        dones = self.dones.reshape(self.episode_length * self.n_env, self.n_agents, -1)
        active_masks = self.active_masks.reshape(
            self.episode_length * self.n_env, self.n_agents, -1
        )
        advantages = self.advantages.reshape(
            self.episode_length * self.n_env, self.n_agents, -1
        )
        returns = self.returns.reshape(
            self.episode_length * self.n_env,  self.n_agents, -1
        )

        total_num = len(obs)
        if isall:
            sample_num = total_num
        else:
            sample_num = min(total_num, self.max_batch_size)

        sample_indices = random.sample(range(total_num), sample_num)

        if self.shuffle_agent_idx:
            agent_indices = random.sample(range(self.n_agents), self.n_agents)
            batch = Transition(
                obs[sample_indices][:, agent_indices],
                value_preds[sample_indices][:, agent_indices],
                actions[sample_indices][:, agent_indices],
                action_masks[sample_indices][:, agent_indices],
                action_logprobs[sample_indices][:, agent_indices],
                orders[sample_indices][:, agent_indices],
                order_logprobs[sample_indices][:, agent_indices],
                rewards[sample_indices][:, agent_indices],
                dones[sample_indices][:, agent_indices],
                active_masks[sample_indices][:, agent_indices],
                advantages[sample_indices][:, agent_indices],
                returns[sample_indices][:, agent_indices],
            )
        else:
            batch = Transition(
                obs[sample_indices],
                value_preds[sample_indices],
                actions[sample_indices],
                action_masks[sample_indices],
                action_logprobs[sample_indices],
                orders[sample_indices],
                order_logprobs[sample_indices],
                rewards[sample_indices],
                dones[sample_indices],
                active_masks[sample_indices],
                advantages[sample_indices],
                returns[sample_indices],
            )
        return batch
