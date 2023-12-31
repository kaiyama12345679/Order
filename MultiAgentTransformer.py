import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical, Normal
from networks import Encoder, Decoder
from buffer import Transition
from valuenorm import ValueNorm
from typing import Tuple


class MultiAgentTransformer(nn.Module):
    def __init__(
        self,
        n_dim: int,
        n_head: int,
        num_layer_encoder: int,
        obs_dim: int,
        action_dim: int,
        num_layer_decoder: int,
        n_agent: int,
        gamma: float,
        clip: float,
        lr: float,
        eps: float,
        entropy_coef: float,
        max_grad_norm: float,
        huber_delta: float,
        device: torch.device,
        discrete = True,
        use_agent_id = False
    ) -> None:
        """
        Initialize MultiAgentTransformer.

        Args:
            n_dim (int): Dimension of the embedding.
            n_head (int): Number of attention heads.
            num_layer_encoder (int): Number of layers for encoder.
            obs_dim (int): Dimension of the observation.
            action_dim (int): Dimension of action space.
            num_layer_decoder (int): Number of layers for decoder.
            n_agent (int): Number of agents.
            gamma (float): Discount factor for rewards.
            clip (float): Clip value for policy gradient.
            lr (float): Learning rate.
            eps (float): Epsilon for optimizer.
            entropy_coef (float): Entropy coefficient for the policy gradient.
            max_grad_norm (float): Maximum gradient norm.
            device (torch.device): The device to run the model on.
        """
        super().__init__()
        self.n_dim = n_dim
        self.device = device
        self.n_head = n_head
        self.num_layer_encoder = num_layer_encoder
        self.num_layer_decoder = num_layer_decoder
        self.device = device
        self.use_agent_id = use_agent_id
        if self.use_agent_id:
            self.encoder = Encoder(n_dim, n_head, obs_dim + n_agent, num_layer_encoder).to(device)
        else:
            self.encoder = Encoder(n_dim, n_head, obs_dim, num_layer_encoder).to(device)
        self.decoder = Decoder(n_dim, n_head, n_agent, action_dim, num_layer_decoder, discrete, use_action_id=False).to(device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=eps)

        self.gamma = gamma
        self.clip = clip
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.huber_delta = huber_delta
        self.n_agent = n_agent

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.discrete = discrete

        # Normal axes should be 2
        self.value_normalizer = ValueNorm(
            input_shape=1, device=self.device
        )

    def get_value(self, state_seq: torch.Tensor) -> torch.Tensor:
        """
        Get the value prediction for given state sequence.

        Args:
            state_seq (torch.Tensor): Tensor of shape (batch_size, seq_len, obs_dim) containing state sequences.

        Returns:
            torch.Tensor: Value prediction for the state sequence.
        """
        if self.use_agent_id:
            state_seq = self._add_id_vector(state_seq)
        hidden_state, values = self.encoder(state_seq)
        return values

    def get_action_and_value(
        self,
        state_seq: torch.Tensor,
        action_mask: torch.Tensor = None,
        deterministic: bool = False,
        action_seq: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action and its value prediction for given state sequence.

        Args:
            state_seq (torch.Tensor): Tensor of shape (batch_size, seq_len, obs_dim) containing state sequences.
            action_mask (torch.Tensor, optional): Action mask. Defaults to None.
            deterministic (bool, optional): Whether to take deterministic actions. Defaults to False.
            action_seq (torch.Tensor): Tensor of shape (batch_size, seq_len, 1) containing action sequences. .

        Returns:
            tuple: Action vector, action log probabilities, entropy, and value prediction for the state sequence.
        """
        n_env, _, _ = state_seq.shape
        if self.use_agent_id:
            state_seq = self._add_id_vector(state_seq)
        hidden_state, values = self.encoder(state_seq)

        if action_seq is None:
            # Recurrent action generation
            action_vector = None
            for i in range(self.n_agent):
                action_logits = self.decoder(action_vector, hidden_state, action_mask)
                latest_action_logit = action_logits[:, i, :].unsqueeze(-2)
                if deterministic:
                    if self.discrete:
                        a = latest_action_logit.argmax(dim=-1).unsqueeze(-1).to(torch.int32)
                    else:
                        a = latest_action_logit
                else:
                    if self.discrete:
                        latest_a = Categorical(logits=latest_action_logit)
                        a = latest_a.sample().unsqueeze(-1).to(torch.int32)
                    else:
                        latest_mean = latest_action_logit
                        action_std = torch.sigmoid(self.decoder.log_std) * 0.5
                        distri = Normal(latest_mean, action_std)
                        a = distri.sample()

                if action_vector is None:
                    action_vector = a
                else:
                    action_vector = torch.cat([action_vector, a], dim=-2)
        else:
            # Action is already provided
            action_vector = action_seq
            action_logits = self.decoder(action_vector, hidden_state, action_mask)
        if self.discrete:
            prob_dist = Categorical(logits=action_logits)
        else:
            action_std = torch.sigmoid(self.decoder.log_std) * 0.5
            prob_dist = Normal(action_logits, action_std)
        # Remove bos
        action_vector = action_vector
        if self.discrete:
            action_logps = prob_dist.log_prob(action_vector.squeeze(-1)).unsqueeze(-1)
            entropy = prob_dist.entropy().unsqueeze(-1)
        else:
            action_logps = prob_dist.log_prob(action_vector)
            entropy = prob_dist.entropy()
        return action_vector, action_logps, entropy, values

    def _add_id_vector(self, state_seq: torch.Tensor):
        batch_size, n_agent, state_dim = state_seq.shape
        id_vector = torch.eye(n_agent).unsqueeze(0).expand(batch_size, -1, -1).to(state_seq.device)
        state_seq = torch.cat([state_seq, id_vector], dim=-1)
        return state_seq

    def update(self, batch: Transition):
        self.train()
        # Model forward
        _, new_action_logps, entropy, new_values = self.get_action_and_value(
            batch.obs, action_mask=batch.action_masks, action_seq=batch.actions
        )

        new_values = new_values.view(-1, 1)
        batch_values = batch.values.view(-1, 1)
        batch_returns = batch.returns.view(-1, 1)
        # update critic
        value_clipped = batch_values + (new_values - batch_values).clamp(
            -self.clip, self.clip
        )

        if self.value_normalizer:
            self.value_normalizer.update(batch_returns)
            normalize_return = self.value_normalizer.normalize(batch_returns)
            critic_loss_clipped = F.huber_loss(
                normalize_return,
                value_clipped,
                reduction="none",
                delta=self.huber_delta,
            )
            critic_loss_original = F.huber_loss(
                normalize_return, new_values, reduction="none", delta=self.huber_delta
            )
        else:
            critic_loss_clipped = F.huber_loss(
                batch_returns,
                value_clipped,
                reduction="none",
                delta=self.huber_delta,
            )
            critic_loss_original = F.huber_loss(
                batch_returns, new_values, reduction="none", delta=self.huber_delta
            )
        _use_clipped_value_loss = True
        if _use_clipped_value_loss:
            critic_loss = torch.max(critic_loss_clipped, critic_loss_original)
        else:
            critic_loss = critic_loss_original

        _use_value_active_masks = True
        if _use_value_active_masks:
            critic_loss = (
                critic_loss * batch.active_masks.view(-1, 1)
            ).sum() / batch.active_masks.sum()
        else:
            critic_loss = critic_loss.mean()
        # critic_loss = (values - returns).pow(2).mean().mean()

        # update actor
        ratio = torch.exp(new_action_logps - batch.action_logprobs)


        # Normalize advantages
        adv = batch.advantages
        normalized_advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

        # mean advantage
        # normalized_advantages = normalized_advantages.mean(dim=-2, keepdim=True)

        surr1 = ratio * normalized_advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * normalized_advantages
        )
        _use_policy_active_masks = True
        if _use_policy_active_masks:
            policy_loss = (
                -torch.sum(torch.min(surr1, surr2) * batch.active_masks)
                / batch.active_masks.sum()
            )

        else:
            policy_loss = -torch.min(surr1, surr2).mean().mean()

        entropy_only_active = (
            entropy * batch.active_masks
        ).sum() / batch.active_masks.sum()
        actor_loss = policy_loss - self.entropy_coef * entropy_only_active

        # Total loss
        loss = critic_loss + actor_loss
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.parameters(), max_norm=self.max_grad_norm
        )
        self.optimizer.step()

        # Additional info for debugging
        # KL
        with torch.no_grad():
            approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
            clip_frac = ((ratio - 1.0).abs() > self.clip).float().mean().item()
            # explained_variance
            if self.value_normalizer:
                denormalized_values = self.value_normalizer.denormalize(batch_values)
                y_pred = denormalized_values.cpu().numpy()
                y_true = batch_returns.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = (
                    np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                )
            else:
                explained_var = 0

        return (
            critic_loss,
            policy_loss,
            grad_norm,
            entropy_only_active,
            ratio,
            approx_kl,
            clip_frac,
            explained_var,
        )

    def load_model(self, path: str) -> None:
        """
        Load model from a file.

        Args:
            path (str): Path to the file.
        """
        self.load_state_dict(torch.load(path + ".pt", map_location=torch.device("cpu")))

    def save_model(self, path: str) -> None:
        """
        Save model to a file.

        Args:
            path (str): Path to save the model.
        """
        torch.save(self.state_dict(), path + ".pt")
