import torch
import torch.nn as nn
import torch.nn.functional as F


def init_(module, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain("relu")
    nn.init.orthogonal_(module.weight.data, gain=gain)
    if module.bias is not None:
        nn.init.constant_(module.bias.data, 0)
    return module


def init_mha_(module, gain=0.01):
    nn.init.constant_(module.in_proj_bias.data, 0.0)
    nn.init.constant_(module.out_proj.bias.data, 0.0)
    nn.init.orthogonal_(module.out_proj.weight.data, gain=gain)
    dim = module.embed_dim
    for i in range(3):
        nn.init.orthogonal_(
            module.in_proj_weight.data[i * dim : (i + 1) * dim], gain=gain
        )


class Encoder(nn.Module):
    def __init__(self, n_dim, n_head, state_dim, num_layer_encoder) -> None:
        super().__init__()

        self.n_dim = n_dim
        self.n_head = n_head
        self.num_layer_encoder = num_layer_encoder

        self.encode_embed = nn.Sequential(
            nn.LayerNorm(state_dim),
            init_(nn.Linear(state_dim, n_dim), activate=True),
            nn.GELU(),
        )

        self.ln = nn.LayerNorm(n_dim)

        self.transformer_encoder = nn.ModuleList(
            [EncoderBlock(n_dim, n_head) for _ in range(num_layer_encoder)]
        )

        self.mlp_encoder = nn.Sequential(
            init_(nn.Linear(n_dim, n_dim), activate=True),
            nn.GELU(),
            nn.LayerNorm(n_dim),
            init_(nn.Linear(n_dim, 1)),
        )

    def forward(self, state_seq):
        """
        state_seq: (batch_size, seq_len, state_dim)
        """
        batch_size, seq_len, state_dim = state_seq.shape

        hidden_state = self.encode_embed(state_seq)
        hidden_state = self.ln(hidden_state)

        for encoder in self.transformer_encoder:
            hidden_state = encoder(hidden_state)

        values = self.mlp_encoder(hidden_state)

        return hidden_state, values


class EncoderBlock(nn.Module):
    def __init__(self, n_dim, n_head):
        super().__init__()
        self.norm1 = nn.LayerNorm(n_dim)
        self.norm2 = nn.LayerNorm(n_dim)
        self.mha = nn.MultiheadAttention(n_dim, n_head, batch_first=True)
        # Init weight
        init_mha_(self.mha)

        self.mlp = nn.Sequential(
            init_(nn.Linear(n_dim, n_dim), activate=True),
            nn.GELU(),
            init_(nn.Linear(n_dim, n_dim)),
        )

    def forward(self, x):
        attn_output, _ = self.mha(x, x, x)
        x = self.norm1(x + attn_output)
        x = self.norm2(x + self.mlp(x))
        return x


class Decoder(nn.Module):
    def __init__(self, n_dim, n_head, action_dim, num_decoder_layer, discrete=True) -> None:
        super().__init__()

        self.n_dim = n_dim
        self.n_head = n_head
        self.action_dim = action_dim
        self.num_decoder_layer = num_decoder_layer

        # self.decode_embed = nn.Embedding(action_dim + 1, n_dim)   # Equivalent except for init and GELU?
        if discrete:
            self.decode_embed = nn.Sequential(
                init_(nn.Linear(action_dim + 1, n_dim, bias=False), activate=True),
                nn.GELU(),
            )
        else:
            self.decode_embed = nn.Sequential(
                init_(nn.Linear(action_dim, n_dim, bias=False), activate=True),
                nn.GELU(),
            )

        self.ln = nn.LayerNorm(n_dim)

        self.decoder = nn.ModuleList(
            [DecoderBlock(n_dim, n_head) for _ in range(num_decoder_layer)]
        )

        self.mlp_decoder = nn.Sequential(
            init_(nn.Linear(n_dim, n_dim), activate=True),
            nn.GELU(),
            nn.LayerNorm(n_dim),
            init_(nn.Linear(n_dim, action_dim)),
        )

        if not discrete:
            self.log_std = nn.Parameter(torch.ones(action_dim))
            self.bos = nn.Paramter(torch.randn(1, 1, action_dim))
        else:
            self.bos = torch.full((1, 1, 1), action_dim)

        self.discrete = discrete

    def forward(self, action_seq, hidden_state, action_mask=None):
        """
        action_seq: (batch_size, seq_len)
        hidden_state: (batch_size, seq_len, n_dim)
        """
        batch_size, n_agent, n_dim = hidden_state.shape
        if action_seq is not None:
            action_seq = torch.concat([self.bos.expand(batch_size, -1, -1), action_seq], dim=-2)
        else:
            action_seq = self.bos.expand(batch_size, -1, -1)
        if self.discrete
            one_hot_action_seq = F.one_hot(action_seq.squeeze(-1), num_classes=self.action_dim + 1)
            action_embed_seq = self.decode_embed(one_hot_action_seq.to(dtype=torch.float))
        else:
            action_embed_seq = self.decode_embed(action_seq)
        action_embed_seq = self.ln(action_embed_seq)

        for decoder in self.decoder:
            action_embed_seq = decoder(tgt=hidden_state, src=action_embed_seq)
        action_logit = self.mlp_decoder(action_embed_seq)
        if action_mask is not None:
            action_logit = action_logit + (1 - action_mask) * (-1e9)

        return action_logit


class DecoderBlock(nn.Module):
    def __init__(self, n_dim, n_head) -> None:
        super().__init__()
        self.n_dim = n_dim
        self.n_head = n_head

        self.mha_self = nn.MultiheadAttention(
            embed_dim=n_dim, num_heads=n_head, batch_first=True
        )
        self.mha_srctgt = nn.MultiheadAttention(
            embed_dim=n_dim, num_heads=n_head, batch_first=True
        )
        # Init weight
        init_mha_(self.mha_self)
        init_mha_(self.mha_srctgt)

        self.norm1 = nn.LayerNorm(n_dim)
        self.norm2 = nn.LayerNorm(n_dim)
        self.norm3 = nn.LayerNorm(n_dim)

        self.feedforward = nn.Sequential(
            init_(nn.Linear(n_dim, n_dim), activate=True),
            nn.GELU(),
            init_(nn.Linear(n_dim, n_dim)),
        )

    def forward(self, tgt, src):
        tgt_len, src_len = tgt.shape[-2], src.shape[-2]
        # self attention mask
        ones = torch.ones(src_len, src_len).to(src.device)
        self_mask = torch.triu(ones, diagonal=1).bool()

        # src-tgt mask
        ones = torch.ones(tgt_len, src_len).to(src.device)
        srctgt_mask = torch.triu(ones, diagonal=1).bool()

        # self attention
        hidden_self, _ = self.mha_self(src, src, src, attn_mask=self_mask)
        hidden_self = self.norm1(hidden_self + src)

        # source-tgt attention
        hidden_srctgt, _ = self.mha_srctgt(
            tgt, hidden_self, hidden_self, attn_mask=srctgt_mask
        )
        hidden_srctgt = self.norm2(hidden_srctgt + tgt)

        # feedforward
        hidden_ff = self.feedforward(hidden_srctgt)
        hidden_ff = self.norm3(hidden_ff + hidden_srctgt)

        return hidden_ff
