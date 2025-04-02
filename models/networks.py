from cmath import sin
from collections import OrderedDict
from dataclasses import dataclass
import imp
import math
from tkinter import E
import torch
import torch.nn as nn
import torch.nn.functional as F
from .clip import clip, tokenize

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

# positional embedding with sin/cos
class Embedder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
        '''
        :param input_dim: dimension of input to be embedded
        :param max_freq_log2: log2 of max freq; min freq is 1 by default
        :param N_freqs: number of frequency bands
        :param log_sampling: if True, frequency bands are linerly sampled in log-space
        :param include_input: if True, raw input is included in the embedding
        :param periodic_fns: periodic functions used to embed input
        '''
        super(Embedder, self).__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(
                2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input: torch.Tensor):
        '''
        :param input: tensor of shape [..., self.input_dim]
        :return: tensor of shape [..., self.out_dim]
        '''
        assert (input.shape[-1] == self.input_dim)

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))
        out = torch.cat(out, dim=-1)

        assert (out.shape[-1] == self.out_dim)
        return out

def get_embedder(multires, input_dim=3):
    if multires < 0:
        return nn.Identity(), input_dim

    embed_kwargs = {
        "include_input": True,  # needs to be True for ray_bending to work properly
        "input_dim": input_dim,
        "max_freq_log2": multires - 1,
        "N_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    return embedder_obj, embedder_obj.out_dim

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, embed_dim, max_len=80):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        po = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.) / embed_dim))
        pe[:, 0::2] = torch.sin(po * div_term)
        pe[:, 1::2] = torch.cos(po * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        l, _, _ = x.shape
        pos = self.pe[:l, :].unsqueeze(1)
        return pos

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, embed_dim, max_len=80, padding_idx=0):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, embed_dim, padding_idx)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.pos_embed.weight)

    def forward(self, x):
        l, _, _ = x.shape
        idx = torch.arange(l, device=x.device)
        pos = self.pos_embed(idx).unsqueeze(1)
        return pos

class RotaryPositionEncoding(nn.Module):
    def __init__(self, feature_dim, pe_type='Rotary1D'):
        super().__init__()

        self.feature_dim = feature_dim
        self.pe_type = pe_type

    @staticmethod
    def embed_rotary(x, cos, sin):
        x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x).contiguous()
        x = x * cos + x2 * sin
        return x

    def forward(self, x_position):
        bsize, npoint = x_position.shape
        div_term = torch.exp(
            torch.arange(0, self.feature_dim, 2, dtype=torch.float, device=x_position.device)
            * (-math.log(10000.0) / (self.feature_dim)))
        div_term = div_term.view(1, 1, -1) # [1, 1, d]

        sinx = torch.sin(x_position * div_term)  # [B, N, d]
        cosx = torch.cos(x_position * div_term)

        sin_pos, cos_pos = map(
            lambda feat: torch.stack([feat, feat], dim=-1).view(bsize, npoint, -1),
            [sinx, cosx]
        )
        position_code = torch.stack([cos_pos, sin_pos] , dim=-1)

        if position_code.requires_grad:
            position_code = position_code.detach()

        return position_code

class RotaryPositionEncoding3D(RotaryPositionEncoding):
    def __init__(self, feature_dim, pe_type='Rotary3D'):
        super().__init__(feature_dim, pe_type)

    @torch.no_grad()
    def forward(self, XYZ):
        '''
        @param XYZ: [B,N,3]
        @return:
        '''
        bsize, npoint, _ = XYZ.shape
        x_position, y_position, z_position = XYZ[..., 0:1], XYZ[..., 1:2], XYZ[..., 2:3]
        div_term = torch.exp(
            torch.arange(0, self.feature_dim // 3, 2, dtype=torch.float, device=XYZ.device)
            * (-math.log(10000.0) / (self.feature_dim // 3))
        )
        div_term = div_term.view(1, 1, -1)  # [1, 1, d//6]

        sinx = torch.sin(x_position * div_term)  # [B, N, d//6]
        cosx = torch.cos(x_position * div_term)
        siny = torch.sin(y_position * div_term)
        cosy = torch.cos(y_position * div_term)
        sinz = torch.sin(z_position * div_term)
        cosz = torch.cos(z_position * div_term)

        sinx, cosx, siny, cosy, sinz, cosz = map(
            lambda feat: torch.stack([feat, feat], -1).view(bsize, npoint, -1),
            [sinx, cosx, siny, cosy, sinz, cosz]
        )

        position_code = torch.stack([
            torch.cat([cosx, cosy, cosz], dim=-1),  # cos_pos
            torch.cat([sinx, siny, sinz], dim=-1)  # sin_pos
        ], dim=-1)

        if position_code.requires_grad:
            position_code = position_code.detach()

        return position_code
    
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

# Self Attention
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward_v(self, x: torch.Tensor):
        """
        Forward function for computing the value features for dense prediction (i.e., features for every image patch).
        """
        # Get the weights and biases for the value projection, multihead attention uses 3 * embed_dim for the input projection
        v_in_proj_weight = self.attn.in_proj_weight[-self.attn.embed_dim:]
        v_in_proj_bias = self.attn.in_proj_bias[-self.attn.embed_dim:]

        v_in = F.linear(self.ln_1(x), v_in_proj_weight, v_in_proj_bias)
        v_out = F.linear(v_in, self.attn.out_proj.weight, self.attn.out_proj.bias)

        # Using the value features works the best. Adding this to 'x' or feeding 'v' to the LayerNorm then MLP degrades the performance
        return v_out


    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

# Cross Attention
class CrossResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = CrossModalAttention(embed_dim=d_model, num_heads=n_head, output_dim=d_model)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor = None):
        self.attn_mask = attn_mask.to(dtype=q.dtype, device=q.device) if attn_mask is not None else None
        attn_output, attn_weights = self.attn(q=q, k=k, v=v, attn_mask=self.attn_mask)
        return attn_output, attn_weights

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor = None):
        attn_output, attn_weights = self.attention(self.ln_1(q), self.ln_1(k), self.ln_1(v), attn_mask)
        q = q + attn_output
        q = q + self.mlp(self.ln_2(q))
        return q, attn_weights

# multi layer
class CrossTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([CrossResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor = None):
        for i, _ in enumerate(self.resblocks):
            q, attn_weights = self.resblocks[i](q, k, v, attn_mask)

        q = q.permute(1, 0, 2) # L'ND -> NL'D
        return q, attn_weights

# one layer without shortcut: naivest cross attention
class CrossModalAttention(nn.Module):
    """ Cross-Modal Attention. Adapted from: https://github.com/openai/CLIP/blob/main/clip/model.py#L56 """

    def __init__(self, embed_dim=1024, num_heads=32, output_dim=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim)

    def forward(self, q, k, v, attn_mask=None):
        x, attn_weights = F.multi_head_attention_forward(
            query=q, key=k, value=v,
            embed_dim_to_check=v.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            need_weights=True,
            attn_mask=attn_mask
        )
        
        return x, attn_weights

# A unified network architecture for grasp and place
class CLIPActionFusion(nn.Module):
    def __init__(self, action_dim, width, layers, heads, device, task_num=None, use_rope=False, no_feat_rope=False, sa=False, no_rgb_feat=False):
        super().__init__()
        
        self.device = device
        
        # cross attention
        self.cross_attn = CrossTransformer(width=width, layers=layers, heads=heads)

        self.sa = sa
        if self.sa:
            self.fusion_attn = Transformer(width=width, layers=layers*2, heads=heads)
        
        hidden_dim = int(width / 2)
        # hidden_dim = 256 
        
        self.action_embbedding = nn.Sequential(
                                nn.Linear(action_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, width),
                                nn.ReLU(),
                                nn.Linear(width, width)
                                )

        self.pos_projection, pos_proj_dim = get_embedder(multires=5, input_dim=3)
        self.point_embbedding = nn.Sequential(
                                nn.Linear(pos_proj_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, width),
                                nn.ReLU(),
                                nn.Linear(width, width)
                                )
        
        # rope layer
        self.use_rope = use_rope
        if self.use_rope:
            self.no_feat_rope = no_feat_rope
            self.relative_pe_layer = RotaryPositionEncoding3D(width)
        
        # whether use rgb
        self.no_rgb_feat = no_rgb_feat
        
        if task_num is not None:
            self.task_embedding = nn.Embedding(task_num + 1, width)
            # self.task_embedding = nn.Parameter(torch.zeros(width))

    def encode_action(self, x):
        grasp_emb = self.action_embbedding(x.to(self.device)) # shape = [N, L', D]
        return grasp_emb

    def encode_pts_pos(self, x):
        pts_pos_emb = self.point_embbedding(x.to(self.device))
        return pts_pos_emb

    def pts_multi_fusion(self, pts_feat, pts_sim):
        fusion_feat = pts_feat * pts_sim
        return fusion_feat        

    def forward(self, pts_pos, pts_feat, pts_sim, actions, mode):
        # get point features weighted by visual-language similarity
        pts_sim_feat = self.pts_multi_fusion(pts_feat, pts_sim)

        # encode point positions
        pts_pos = self.pos_projection(pts_pos)
        pts_pos_feat = self.encode_pts_pos(pts_pos)
        
        if self.use_rope:
            # add rope
            pos_rotaty_pe = self.relative_pe_layer(pts_pos.to(self.device))
            pos_cos, pos_sin = pos_rotaty_pe[..., 0], pos_rotaty_pe[..., 1]
            
            if not self.no_feat_rope:
                pts_sim_feat = RotaryPositionEncoding.embed_rotary(pts_sim_feat, pos_cos, pos_sin)
            pts_pos_feat = RotaryPositionEncoding.embed_rotary(pts_pos_feat, pos_cos, pos_sin)
        
        pts_sim_feat = pts_sim_feat.permute(1, 0, 2) # NLD -> LND
        pts_pos_feat = pts_pos_feat.permute(1, 0, 2) # NLD -> LND
        
        action_feat = self.encode_action(actions) # shape = [N, L', D]
            
        action_feat = action_feat.permute(1, 0, 2) # NL'D -> L'ND
        
        # add task embedding
        if mode is not None:
            if mode == "grasp":
                task_feat = self.task_embedding(torch.LongTensor([1]).to(self.device)).repeat(action_feat.shape[0], 1, 1)
            elif mode == "place":
                task_feat = self.task_embedding(torch.LongTensor([2]).to(self.device)).repeat(action_feat.shape[0], 1, 1)
            
            action_feat = action_feat + task_feat
        
        # cross-attention
        if not self.no_rgb_feat:
            cross_feat, attn_weights = self.cross_attn(q=action_feat, k=pts_pos_feat, v=pts_sim_feat) # shape = [N, L', D], [N, L', L]
        else:
            cross_feat, attn_weights = self.cross_attn(q=action_feat, k=pts_pos_feat, v=pts_pos_feat) # shape = [N, L', D], [N, L', L]

        # self-attention
        if self.sa:
            cross_feat = cross_feat.permute(1, 0, 2) # NL'D -> L'ND
            cross_feat = self.fusion_attn(cross_feat) # shape = [L', N, D]
            cross_feat = cross_feat.permute(1, 0, 2) # L'ND -> NL'D

        return cross_feat, attn_weights

# A unified network architecture for grasp and place
class CLIPActionLangFusion(nn.Module):
    def __init__(self, action_dim, width, layers, heads, lang_enc, device, task_num=None, use_rope=False):
        super().__init__()
        
        self.device = device
        
        # cross attention
        self.lang_cross_attn = CrossTransformer(width=width, layers=layers, heads=heads)
        self.cross_attn = CrossTransformer(width=width, layers=layers, heads=heads)
        
        hidden_dim = int(width / 2)
        # hidden_dim = 256 
        
        self.action_embbedding = nn.Sequential(
                                nn.Linear(action_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, width),
                                nn.ReLU(),
                                nn.Linear(width, width)
                                )

        self.pos_projection, pos_proj_dim = get_embedder(multires=5, input_dim=3)
        self.point_embbedding = nn.Sequential(
                                nn.Linear(pos_proj_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, width),
                                nn.ReLU(),
                                nn.Linear(width, width)
                                )
        
        # rope layer
        self.use_rope = use_rope
        if self.use_rope:
            self.relative_pe_layer = RotaryPositionEncoding3D(width)
        
        self.lang_enc = lang_enc
        if "clip" in self.lang_enc:
            self.clip_lang_encoder, _ = clip.load('ViT-L/14@336px', device=self.device)
                        
        if task_num is not None:
            # self.task_embedding = nn.Embedding(task_num + 1, width)
            self.task_embedding = nn.Parameter(torch.zeros(width))

    def encode_text(self, x):
        with torch.no_grad():
            if "clip" in self.lang_enc:
                tokens = tokenize(x).to(self.device)
                if "full" in self.lang_enc:
                    text_emb = self.clip_lang_encoder.encode_text_full(tokens)  # shape = [N, 77, D]
                else:
                    text_emb = self.clip_lang_encoder.encode_text(tokens).unsqueeze(0)  # shape = [N, 1, D]
        return text_emb
            
    def encode_action(self, x):
        grasp_emb = self.action_embbedding(x.to(self.device)) # shape = [N, L', D]
        return grasp_emb

    def encode_pts_pos(self, x):
        pts_pos_emb = self.point_embbedding(x.to(self.device))
        return pts_pos_emb

    def pts_multi_fusion(self, pts_feat, pts_sim):
        fusion_feat = pts_feat * pts_sim
        return fusion_feat        

    def forward(self, pts_pos, pts_feat, actions, lang_goal, mode):
        
        # encode language feature
        lang_feat = self.encode_text(lang_goal).to(torch.float32)
        # get point features conditioned on language feature
        lang_feat = lang_feat.permute(1, 0, 2) # NLD -> LND
        pts_feat = pts_feat.permute(1, 0, 2) # NLD -> LND
        pts_sim_feat, _ = self.lang_cross_attn(q=pts_feat, k=lang_feat, v=lang_feat)

        # encode point positions
        pts_pos = self.pos_projection(pts_pos)
        pts_pos_feat = self.encode_pts_pos(pts_pos)
        
        if self.use_rope:
            # add rope
            pos_rotaty_pe = self.relative_pe_layer(pts_pos.to(self.device))
            pos_cos, pos_sin = pos_rotaty_pe[..., 0], pos_rotaty_pe[..., 1]
            
            pts_sim_feat = RotaryPositionEncoding.embed_rotary(pts_sim_feat, pos_cos, pos_sin)
            pts_pos_feat = RotaryPositionEncoding.embed_rotary(pts_pos_feat, pos_cos, pos_sin)
        
        pts_sim_feat = pts_sim_feat.permute(1, 0, 2) # NLD -> LND
        pts_pos_feat = pts_pos_feat.permute(1, 0, 2) # NLD -> LND
       
        action_feat = self.encode_action(actions) # shape = [N, L', D]
            
        action_feat = action_feat.permute(1, 0, 2) # NL'D -> L'ND
        
        # add task embedding
        if mode is not None:
            action_feat = action_feat + self.task_embedding
        
        # cross-attention
        cross_feat, attn_weights = self.cross_attn(q=action_feat, k=pts_pos_feat, v=pts_sim_feat) # shape = [N, L', D], [N, L', L]

        return cross_feat, attn_weights

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

# Policy Network for RL algorithms
class Policy(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_norm=False):
        super(Policy, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

        # layer normalization
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)
            
    def forward(self, state):
        if self.layer_norm:
            x = F.relu(self.ln1(self.linear1(state)))
            x = F.relu(self.ln2(self.linear2(x)))
            logits = self.linear3(x).squeeze()
        else:
            x = F.relu(self.linear1(state))
            x = F.relu(self.linear2(x))
            logits = self.linear3(x).squeeze()
        return logits
  
class CLIPAction(nn.Module):
    def __init__(self, action_dim, args):
        super().__init__()
        self.device = args.device
        self.vilg_fusion = CLIPActionFusion(action_dim, args.width, args.layers, args.heads, self.device, args.task_num, args.use_rope, args.no_feat_rope, args.fusion_sa, args.no_rgb_feat).to(device=self.device)
        self.policy = Policy(args.width, args.hidden_size, args.layer_norm).to(self.device)

    def forward(self, pts_pos, pts_feat, pts_sim, actions, mode=None):
        vilg_feature, _ = self.vilg_fusion(pts_pos, pts_feat, pts_sim, actions, mode)
        logits = self.policy(vilg_feature)
        if vilg_feature.shape[0] == 1:
            logits = logits.unsqueeze(0)
        if actions.shape[1] == 1:
            logits = logits.unsqueeze(0)
        action = logits.argmax(-1)  # [B,]

        return logits, action
    
class CLIPLangEmbAction(nn.Module):
    def __init__(self, action_dim, args):
        super().__init__()
        self.device = args.device
        self.vilg_fusion = CLIPActionLangFusion(action_dim, args.width, args.layers, args.heads, args.lang_enc, self.device, args.task_num, args.use_rope).to(device=self.device)
        self.policy = Policy(args.width, args.hidden_size).to(self.device)

    def forward(self, pts_pos, pts_feat, actions, lang_goal, mode=None):
        vilg_feature, _ = self.vilg_fusion(pts_pos, pts_feat, actions, lang_goal, mode)
        logits = self.policy(vilg_feature)
        if vilg_feature.shape[0] == 1:
            logits = logits.unsqueeze(0)
        if actions.shape[1] == 1:
            logits = logits.unsqueeze(0)
        action = logits.argmax(-1)  # [B,]

        return logits, action
    
class AdaptPolicyCLIPAction(nn.Module):
    def __init__(self, action_dim, args):
        super().__init__()
        self.device = args.device
        self.vilg_fusion = CLIPActionFusion(action_dim, args.width, args.layers, args.heads, self.device, args.task_num, args.use_rope, args.fusion_sa).to(device=self.device)
        self.policy = Policy(args.width, args.hidden_size).to(self.device)
        self.residual_policy = Policy(args.width, args.hidden_size).to(self.device)

    def forward(self, pts_pos, pts_feat, pts_sim, actions, mode=None, ratio=0.5):
        vilg_feature, _ = self.vilg_fusion(pts_pos, pts_feat, pts_sim, actions, mode)
        base_logits = self.policy(vilg_feature)
        residual_logits = self.residual_policy(vilg_feature)
        
        logits = (1 - ratio) * base_logits + ratio * residual_logits
        
        if vilg_feature.shape[0] == 1:
            logits = logits.unsqueeze(0)
        if actions.shape[1] == 1:
            logits = logits.unsqueeze(0)
        action = logits.argmax(-1)  # [B,]

        return logits, action
    
class AdaptFeatCLIPAction(nn.Module):
    def __init__(self, action_dim, args):
        super().__init__()
        self.device = args.device
        self.vilg_fusion = CLIPActionFusion(action_dim, args.width, args.layers, args.heads, self.device, args.task_num, args.use_rope, args.fusion_sa).to(device=self.device)
        self.feat_adapter = Adapter(args.width).to(device=self.device)
        self.policy = Policy(args.width, args.hidden_size).to(self.device)

    def forward(self, pts_pos, pts_feat, pts_sim, actions, mode=None, ratio=0.2):
        vilg_feature, _ = self.vilg_fusion(pts_pos, pts_feat, pts_sim, actions, mode)
        adapt_feature = self.feat_adapter(vilg_feature)
        vilg_adapt_feature = (1 - ratio) * vilg_feature + ratio * adapt_feature

        logits = self.policy(vilg_adapt_feature)
        
        if vilg_adapt_feature.shape[0] == 1:
            logits = logits.unsqueeze(0)
        if actions.shape[1] == 1:
            logits = logits.unsqueeze(0)
        action = logits.argmax(-1)  # [B,]

        return logits, action