import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical, Normal
from networks import Encoder, Decoder, Pointer, OrderedEncoder, positional_encoding, Transformer_Pointer
from buffer import Transition
from valuenorm import ValueNorm
from typing import Tuple
import math
import random


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
        self.encoder = Encoder(n_dim, n_head, obs_dim + n_agent, num_layer_encoder).to(device)
        self.decoder = Decoder(n_dim, n_head, n_agent, action_dim, num_layer_decoder, discrete).to(device)

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
        self.order_clip = 0.2

        # Normal axes should be 2
        self.value_normalizer = ValueNorm(
            input_shape=1, norm_axes=2, device=self.device
        )

    def get_value(self, state_seq: torch.Tensor) -> torch.Tensor:
        """
        Get the value prediction for given state sequence.

        Args:
            state_seq (torch.Tensor): Tensor of shape (batch_size, seq_len, obs_dim) containing state sequences.

        Returns:
            torch.Tensor: Value prediction for the state sequence.
        """
        state_seq = self._add_id_vector(state_seq)
        hidden_state, values = self.encoder(state_seq)
        return values

    def get_action_and_value(
        self,
        state_seq: torch.Tensor,
        action_mask: torch.Tensor = None,
        deterministic: bool = False,
        action_seq: torch.Tensor = None,
        order_seq: torch.Tensor = None,
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
        n_env, n_agent, _ = state_seq.shape
        state_seq = self._add_id_vector(state_seq)
        hidden_state, values = self.encoder(state_seq)

        ordered_state = None
        ordered_action_mask = action_mask

        order = None
        if order_seq is None:
            action_vector = None
            order = torch.stack([torch.randperm(n_agent) for _ in range(n_env)]).to(self.device)
            ordered_state = torch.gather(
                    hidden_state,
                    dim=-2,
                    index=order.unsqueeze(-1).expand(-1, -1, hidden_state.shape[-1]),
                )
            ordered_action_mask = torch.gather(
                    action_mask,
                    dim=-2,
                    index=order.unsqueeze(-1).expand(-1, -1, action_mask.shape[-1]),
                )
            for i in range(n_agent):
                action_logits = self.decoder(action_vector, order, hidden_state, ordered_state, ordered_action_mask)
                latest_action_logit = action_logits[:, -1, :].unsqueeze(-2)
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
            if len(order_seq.shape) == 3:
                order_seq = order_seq.squeeze(-1)
            order_seq = order_seq.to(torch.int64)
            action_vector = action_seq
            action_vector = torch.gather(
                action_vector,
                dim=-2,
                index=order_seq.unsqueeze(-1).expand(-1, -1, action_vector.shape[-1]),
            )
            ordered_state = torch.gather(
                hidden_state,
                dim=-2,
                index=order_seq.unsqueeze(-1).expand(-1, -1, hidden_state.shape[-1]),
            )
            ordered_action_mask = torch.gather(
                    action_mask,
                    dim=-2,
                    index=order_seq.unsqueeze(-1).expand(-1, -1, action_mask.shape[-1]),
                )
            action_logits = self.decoder(action_vector, order_seq, hidden_state, ordered_state, ordered_action_mask)
            order = order_seq
        
        order_logprobs = torch.ones((n_env, n_agent, 1)).to(state_seq.device)

        if self.discrete:
            prob_dist = Categorical(logits=action_logits)
        else:
            action_std = torch.sigmoid(self.decoder.log_std) * 0.5
            prob_dist = Normal(action_logits, action_std)
        # Remove bos
        action_vector = action_vector

        reversed_index = torch.argsort(order, dim=-1)
        if self.discrete:
            action_logps = prob_dist.log_prob(action_vector.squeeze(-1)).unsqueeze(-1)
            entropy = prob_dist.entropy().unsqueeze(-1)
        else:
            action_logps = prob_dist.log_prob(action_vector)
            entropy = prob_dist.entropy()
        
        action_vector = torch.gather(
            action_vector,
            dim=-2,
            index=reversed_index.unsqueeze(-1).expand(-1, -1, action_vector.shape[-1]),
        )
        return action_vector, action_logps, entropy, \
                order.unsqueeze(-1), order_logprobs, 1, values
    
    def _add_id_vector(self, state_seq: torch.Tensor):
        batch_size, n_agent, state_dim = state_seq.shape
        id_vector = torch.eye(n_agent).unsqueeze(0).expand(batch_size, -1, -1).to(state_seq.device)
        state_seq = torch.cat([state_seq, id_vector], dim=-1)
        return state_seq

    def update(self, batch: Transition, beta=0):
        self.train()
        # temp = random.randint(0, 1)
        temp = "REINFORCE"
        # Model forward
        _, new_action_logps, entropy, _, new_order_logprobs, order_entropy, new_values = self.get_action_and_value(
            batch.obs, action_mask=batch.action_masks, action_seq=batch.actions, order_seq=batch.orders
        )

        # update critic
        value_clipped = batch.values + (new_values - batch.values).clamp(
            -self.clip, self.clip
        )

        if self.value_normalizer:
            self.value_normalizer.update(batch.returns)
            normalize_return = self.value_normalizer.normalize(batch.returns)
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
                batch.returns,
                value_clipped,
                reduction="none",
                delta=self.huber_delta,
            )
            critic_loss_original = F.huber_loss(
                batch.returns, new_values, reduction="none", delta=self.huber_delta
            )
        _use_clipped_value_loss = True
        if _use_clipped_value_loss:
            critic_loss = torch.max(critic_loss_clipped, critic_loss_original)
        else:
            critic_loss = critic_loss_original

        _use_value_active_masks = True
        if _use_value_active_masks:
            critic_loss = (
                critic_loss * batch.active_masks
            ).sum() / batch.active_masks.sum()
        else:
            critic_loss = critic_loss.mean()
        order_ratio = torch.exp(new_order_logprobs - batch.order_logprobs)
        n_agent = new_action_logps.shape[-2]
        adv = batch.advantages
        normalized_advantages = (adv - adv.mean()) / (adv.std() + 1e-8)
        normalized_advantages = normalized_advantages.mean(dim=-2, keepdim=True)
        ratio = torch.exp(new_action_logps - batch.action_logprobs)
        surr1 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * normalized_advantages
        surr2 = ratio * normalized_advantages

        _use_policy_active_masks = True
        ordered_active_masks = torch.gather(
            batch.active_masks,
            dim=-2,
            index=batch.orders.to(torch.int64))
        if _use_policy_active_masks:
            policy_loss = (
                -(torch.sum(torch.min(surr1, surr2) * ordered_active_masks))
                 / (ordered_active_masks.sum())
            )

        else:
            policy_loss = (
                -torch.min(surr1, surr2).mean()
            )
        
        entropy_only_active = (
            entropy * ordered_active_masks
        ).sum() / ordered_active_masks.sum()
        actor_loss = policy_loss - self.entropy_coef * entropy_only_active

        # Total loss
        self.optimizer.zero_grad()
        loss = critic_loss + actor_loss
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
                denormalized_values = self.value_normalizer.denormalize(batch.values)
                y_pred = denormalized_values.cpu().numpy()
                y_true = batch.returns.cpu().numpy()
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
    
    def calc_orderprob(self, prob: torch.Tensor, index: torch.Tensor):
        batch_size, n_agent, _ = prob.shape
        output_prob = torch.zeros((batch_size, n_agent, n_agent)).to(prob.device)
        comp_prob = 1 - prob
        accum_comp_prob = torch.ones((batch_size, n_agent + 1, n_agent)).to(prob.device)
        for i in range(n_agent):
            accum_comp_prob[:, i+1, :] = accum_comp_prob[:, i, :].clone() * comp_prob[:, i, :]
            #accum_com_prob[i] & prob[i]
        
        output_prob = accum_comp_prob[: :-1, :] * prob
        return output_prob

            


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

def gen_clipvalue(n_agent: int, alpha: float, device: torch.device, step=-1):
    assert alpha >= 0
    if step == -1:
        clips = (torch.arange(start=n_agent, end=0, step=-1).unsqueeze(-1).unsqueeze(0) / n_agent).pow(alpha).to(device)
    elif step == 1:
        clips = (torch.arange(start=1, end=n_agent + 1, step=1).unsqueeze(-1).unsqueeze(0) / n_agent).pow(alpha).to(device)
    else:
        raise ValueError("step value is invalid.")
    return clips    