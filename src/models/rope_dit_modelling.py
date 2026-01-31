import torch
from torch import nn
import torch.nn.functional as F
import math

"""
This is a wireframe for DiT with RoPE embeddings and adaLN-Zero

"""

#############################################
######            Embedders           #######
#############################################

class TimestepEmbedder(nn.Module):
    def __init__(self, model_dim : int, frequency_embedding_size : int=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.adapter = nn.Sequential(
            nn.Linear(self.frequency_embedding_size, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim)
        )
    
    @staticmethod
    def timestep_embedding(t : torch.Tensor , dim : int, max_period : int=10000):
        """
        :param t: timesteps (bs,)
        :param dim: frequency embedding dim
        :return: timesteps_embedded (bs, dim)
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        theta = t.unsqueeze(-1).float() * freqs.unsqueeze(0)
        embeddings = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)
        if dim%2: 
            #ensure dimensions even.
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])],  dim=-1)
        return embeddings
    
    def forward(self, t : torch.Tensor, **kwargs):
        t_freqs = self.timestep_embedding(t, self.frequency_embedding_size, **kwargs)
        timestep_embeddings = self.adapter(t_freqs)
        return timestep_embeddings



class LabelEmbedder(nn.Module):
    def __init__(self, num_classes : int, model_dim : int, use_cfg : bool):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + use_cfg, model_dim)
        self.num_classes = num_classes
        self.model_dim = model_dim
        self.use_cfg = use_cfg

    def forward(self, labels : torch.LongTensor, train : bool):
        """
        :param labels: (bs, )
        """
        label_embeddings = self.embedding_table(labels)
        return label_embeddings    

class Patchifier(nn.Module):
    def __init__(self, model_dim : int, patch_size : int, in_channels : int):
        super().__init__()
        # (bs, C, H, W)
        # (bs, tok, C*ps*ps)
        # (bs, tok, model_dim)

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.model_dim = model_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_channels * patch_size * patch_size, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, in_channels * patch_size * patch_size)
        )

    def forward(self, x : torch.Tensor):
        x = F.unfold(x, self.patch_size, stride=self.patch_size).transpose(1, 2).contiguous()
        x = self.encoder(x)
        return x
    
    def revert(self, x, image_size):
        x = self.decoder(x)
        x = x.transpose(1, 2).contiguous()
        x = F.fold(x, image_size, self.patch_size, stride=self.patch_size)
        return x


def scale_and_shift(x : torch.Tensor, scale : torch.Tensor, shift : torch.Tensor):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

############################################
########        Transformer     ############
############################################

class RoPE2DMHA(nn.Module):
    def __init__(self, model_dim : int, num_attn_heads : int):

        assert (model_dim // num_attn_heads) % 4 == 0, "attn_dim should be multiple of 4 for 2D RoPE"

        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_attn_heads
        self.attn_dim = model_dim // num_attn_heads

        self.Wq = nn.Linear(model_dim, model_dim, bias=True)
        self.Wk = nn.Linear(model_dim, model_dim, bias=True)
        self.Wv = nn.Linear(model_dim, model_dim, bias=True)
        self.outproj = nn.Linear(model_dim, model_dim)

    def apply_rope(self, W : torch.Tensor, cos : torch.Tensor, sin : torch.Tensor):
        """
        :param W: (bs, num_attn_head, num_tokens, attn_dim)
        :param cos: (num_tokens(rows, cols), attn_dim(axes, num_pairs))
        :param sin: (num_tokens(rows, cols), attn_dim(axes, num_pairs))

        :return W: (bs, num_attn_head, num_tokens, attn_dim)
        """

        Wsize = W.shape 

        num_rows, num_cols, num_axes, num_pairs = cos.shape

        W = W.view(Wsize[0], Wsize[1], num_rows, num_cols, num_axes, num_pairs, 2)
        cos = cos.unsqueeze(0).unsqueeze(0).to(device=W.device)
        sin = sin.unsqueeze(0).unsqueeze(0).to(device=W.device)

        W = torch.stack(
            [cos * W[..., 0] - sin * W[..., 1], 
            sin * W[..., 0] + cos * W[..., 1]],
            dim=-1
        )

        W = W.view(*(Wsize))

        return W

    def forward(self, x : torch.Tensor, cos : torch.Tensor, sin : torch.Tensor):
        """
        :param x: (bs, num_tokens, model_dim)
        :return x: (bs, num_tokens, model_dim)
        """

        bs, num_tokens, _ = x.shape

        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)

        Q = Q.view(bs, num_tokens, self.num_heads, self.attn_dim).transpose(1, 2)
        K = K.view(bs, num_tokens, self.num_heads, self.attn_dim).transpose(1, 2)
        V = V.view(bs, num_tokens, self.num_heads, self.attn_dim).transpose(1, 2)

        Q = self.apply_rope(Q, cos, sin)
        K = self.apply_rope(K, cos, sin)

        x = torch.softmax(Q @ K.transpose(-2, -1) / math.sqrt(self.attn_dim), dim=-1) @ V

        x = x.transpose(1, 2).contiguous().view(bs, num_tokens, -1)

        x = self.outproj(x)

        return x

class DiTBlock(nn.Module):
    def __init__(self, model_dim : int,  num_attn_heads : int):
        super().__init__()

        self.model_dim = model_dim
        self.num_attn_heads = num_attn_heads

        self.attn = RoPE2DMHA(model_dim, num_attn_heads)
        self.adaLN = nn.Sequential(
            nn.Linear(2 * model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, 6 * model_dim)
        )
        self.pointwiseffn = nn.Sequential(
            nn.Linear(model_dim, 2*model_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(2*model_dim, model_dim)
        )
    
    def forward(self, x : torch.Tensor, c : torch.Tensor, cos : torch.Tensor, sin : torch.Tensor):

        res = x #(bs, tok, dim)

        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaLN(c).chunk(6, dim=1) #(bs, dim)

        x = F.layer_norm(x, [self.model_dim])

        x = scale_and_shift(x, gamma1, beta1)

        x = alpha1.unsqueeze(1) * self.attn(x, cos, sin)

        x = x + res
        res = x

        x = F.layer_norm(x, [self.model_dim])

        x = scale_and_shift(x, gamma2, beta2)

        x = alpha2.unsqueeze(1) * self.pointwiseffn(x)

        x = x + res

        return x

class RoPEDiT(nn.Module):
    def __init__(self, model_dim : int, num_dit_blocks : int, num_attn_heads : int, 
                 patch_size : int, num_classes : int, in_channels : int = 3, use_cfg = True):
        super().__init__()

        self.patch_size = patch_size
        self.model_dim = model_dim
        self.num_dit_blocks = num_dit_blocks
        self.num_attn_heads = num_attn_heads
        self.num_classes = num_classes
        self.use_cfg = use_cfg

        self.rope_cache = {}

        self.tokenizer = Patchifier(model_dim, patch_size, in_channels)
        self.time_embedder = TimestepEmbedder(model_dim)
        self.label_embedder = LabelEmbedder(num_classes, model_dim, use_cfg)

        self.dit_blocks = nn.ModuleList([
            DiTBlock(model_dim, num_attn_heads) for _ in range(num_dit_blocks)
        ])

        self.initialize_weights()

    def initialize_weights(self):
        
        for block in self.dit_blocks:
            nn.init.zeros_(block.adaLN[-1].weight)
            nn.init.zeros_(block.adaLN[-1].bias)
        

    def compute_rope(self, image_size : tuple[int, int]):
        """
        :param image_size: (H, W)
        """

        assert image_size[0]%self.patch_size == 0, "Image height must be divisible by patch_size"
        assert image_size[1]%self.patch_size == 0, "Image width must be divisible by patch_size"

        num_rows, num_columns = image_size[0] // self.patch_size, image_size[1] // self.patch_size
        num_pairs = (self.model_dim // self.num_attn_heads) // 4
        attn_dim = self.model_dim // self.num_attn_heads

        rows, cols = torch.meshgrid(
            torch.arange(num_rows), torch.arange(num_columns), indexing='ij'
        )  #(rows, cols)

        freqs = 10_000 ** (
            - 2 * torch.arange(num_pairs) / (attn_dim / 2)
        ) #(num_pairs)

        rows_freqs = rows.unsqueeze(-1) * freqs #iA (rows, cols, num_pairs)
        cols_freqs = cols.unsqueeze(-1) * freqs #jA (rows, cols, num_pairs)

        cos_tensor = torch.stack(
            [torch.cos(rows_freqs),
            torch.cos(cols_freqs)],
            dim = 2
        ) # (rows, cols, axes, num_pairs)

        sin_tensor = torch.stack(
            [torch.sin(rows_freqs),
            torch.sin(cols_freqs)],
            dim = 2
        ) # (rows, cols, axes, num_pairs)

        self.rope_cache[image_size] = (cos_tensor, sin_tensor)

    def forward(self, x : torch.Tensor, t : torch.Tensor, y : torch.LongTensor):
        """
        forward pass in RoPEDiT

        """

        image_size = tuple(x.shape[-2:])

        if self.rope_cache.get(image_size, None) is None:
            self.compute_rope(image_size)

        cos, sin = self.rope_cache.get(image_size)

        t_embed = self.time_embedder(t)
        y_embed = self.label_embedder(y, self.training)

        c = torch.cat([t_embed, y_embed], dim=-1)

        x = self.tokenizer(x)

        for block in self.dit_blocks:
            x = block(x, c, cos, sin)

        x = self.tokenizer.revert(x, image_size)

        return x
    
    def forwardCFG(self, x : torch.Tensor, t : torch.Tensor, y : torch.LongTensor, cfg_scale : float = 1):

        assert self.use_cfg, "CFG was not enabled during init"

        y_null = torch.full_like(y, self.num_classes)

        cond = self(x, t, y)
        uncond = self(x, t, y_null)

        return cfg_scale*cond + (1 - cfg_scale)*uncond

