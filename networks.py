import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    def __init__(self, n_dim, n_head, is_causal=False) -> None:
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
        self.is_causal = is_causal

    def forward(self, x):
        if self.is_causal:
            ones = torch.ones(x.shape[1], x.shape[1]).to(x.device)
            self_mask = torch.triu(ones, diagonal=1).bool()
            attn_output, _ = self.mha(x, x, x, attn_mask=self_mask)
        else:
            attn_output, _ = self.mha(x, x, x)
        x = self.norm1(x + attn_output)
        x = self.norm2(x + self.mlp(x))
        return x


class Decoder(nn.Module):
    def __init__(self, n_dim, n_head, action_dim, num_decoder_layer) -> None:
        super().__init__()

        self.n_dim = n_dim
        self.n_head = n_head
        self.action_dim = action_dim
        self.num_decoder_layer = num_decoder_layer

        # self.decode_embed = nn.Embedding(action_dim + 1, n_dim)   # Equivalent except for init and GELU?
        self.decode_embed = nn.Sequential(
            init_(nn.Linear(action_dim + 1, n_dim, bias=False), activate=True),
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

    def forward(self, action_seq, hidden_state, action_mask=None):
        """
        action_seq: (batch_size, seq_len)
        hidden_state: (batch_size, seq_len, n_dim)
        """
        one_hot_action_seq = F.one_hot(action_seq, num_classes=self.action_dim + 1)
        action_embed_seq = self.decode_embed(one_hot_action_seq.to(dtype=torch.float))
        batch_size, seq_len, action_dim = action_embed_seq.shape
        pe = positional_encoding(seq_len, action_dim, hidden_state.device)
        action_embed_seq = self.ln(action_embed_seq) * math.sqrt(action_dim) + pe
        #action_embed_seq = self.ln(action_embed_seq)

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


class Pointer(nn.Module):
    def __init__(self, n_dim):
        super().__init__()
        self.mha_enc = nn.MultiheadAttention(n_dim, num_heads=1, batch_first=True)
        self.mha = nn.MultiheadAttention(n_dim, num_heads=1, batch_first=True)
        self.vl = nn.Parameter(torch.randn(1, 1, n_dim))
        self.Wq = nn.Linear(3 * n_dim, n_dim, bias=False)
        init_mha_(self.mha)
        init_mha_(self.mha_enc)
        


    def forward(self, state_seq, ordered_seq=None, index_seq=None):
        batch_size, n_agent, n_dim = state_seq.shape
        length = index_seq.shape[-1] if (index_seq is not None) else 0

        prob_mask = torch.zeros((batch_size, 1, n_agent), device=state_seq.device)
        if index_seq is not None:
            for i in range(index_seq.shape[1]):
                latest_mask = prob_mask[:, -1, :] + F.one_hot(
                    index_seq[:, i], num_classes=n_agent
                )
                prob_mask = torch.cat([prob_mask, latest_mask.unsqueeze(1)], dim=-2)
        prob_mask = prob_mask.to(torch.bool)
        
        mean_state = torch.mean(state_seq, dim=-2, keepdim=True)
        if ordered_seq is None:
            v = self.vl.expand(batch_size, -1, -1)
        else:
            v = torch.concat([self.vl.expand(batch_size, -1, -1), ordered_seq], dim=-2)
        f = positional_encoding(length + 1, n_dim, state_seq.device).expand(batch_size, -1, -1)

        h = torch.concat([mean_state.expand(-1, length + 1, -1), v, f], dim=-1)
        q0 = self.Wq(h)
        q1, _ = self.mha_enc(q0, state_seq, state_seq, attn_mask = prob_mask)
        _, prob = self.mha(q1, state_seq, state_seq, attn_mask = prob_mask)
        return prob
        
        
    

class OrderedEncoder(nn.Module):
    def __init__(self, n_dim, is_causal=True):
        super().__init__()
        self.encoder = EncoderBlock(n_dim, n_head=1, is_causal=is_causal)
    
    def forward(self, state_seq: torch.Tensor, ordered_seq=None):
        batch_size, n_agent, seq_len = state_seq.shape
        
        net_input = ordered_seq
        if net_input is not None:
            pe = positional_encoding(net_input.shape[1], net_input.shape[-1], net_input.device)
            net_input = math.sqrt(net_input.shape[-1]) * net_input + pe
            x = self.encoder(net_input)
            return x
        else:
            return None




        
    

def positional_encoding(seq_len, d_model, device: torch.device):
    pe = torch.zeros(seq_len, d_model)

    # 位置情報の生成
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

    # 2iの部分にはsinを、2i+1の部分にはcosを適用するための係数を計算
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
    )

    # sinとcosの計算
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0).to(device)

    return pe
