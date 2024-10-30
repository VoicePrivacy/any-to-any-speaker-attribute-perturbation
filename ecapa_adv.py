# Copyright xmuspeech (Author: Leo 2022-05-27)
# refs:
# 1.  ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification
#           https://arxiv.org/abs/2005.07143

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import yaml
import numpy
import torchaudio.compliance.kaldi
import torch.distributed as dist
from torch.distributed import ReduceOp
sys.path.insert(0, 'subtools/pytorch')
import processor

def _pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,         
    missingg_keys,
    unexpected_keys,
    error_msgs,
):
    """Perform pre-hook in load_state_dict for backward compatibility.

    Note:
        We saved self.pe until v.0.5.2 but we have omitted it later.
        Therefore, we remove the item "pe" from `state_dict` for backward compatibility.

    """
    k = prefix + "pe"
    if k in state_dict:
        state_dict.pop(k)
class PositionalEncoding(torch.nn.Module):
    """Positional encoding.

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
        reverse (bool): Whether to reverse the input position. Only for
        the class LegacyRelPositionalEncoding. We remove it in the current
        class RelPositionalEncoding.
    """

    def __init__(self, d_model, dropout_rate=0.0, max_len=5000, reverse=False):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))
        self._register_load_state_dict_pre_hook(_pre_hook)

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model)
        if self.reverse:
            position = torch.arange(
                x.size(1) - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, : x.size(1)]
        return self.dropout(x)

class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.

    Args:
        nout (int): Output dim size.
        dim (int): Dimension to be normalized.

    """

    def __init__(self, nout, dim=-1):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.

        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return (
            super(LayerNorm, self)
            .forward(x.transpose(self.dim, -1))
            .transpose(self.dim, -1)
        )

class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_head,n_att, n_feat, dropout_rate=0.0):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_att % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_att // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_att)
        self.linear_k = nn.Linear(n_feat, n_att)
        self.linear_v = nn.Linear(n_feat, n_att)
        self.linear_out = nn.Linear(n_feat, n_att)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            self.attn = torch.softmax(scores, dim=-1)
            # mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            # # min_value = float(
            # #     numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min
            # # )
            # # scores = scores.masked_fill(mask, min_value)
            # self.attn = torch.softmax(scores, dim=-1).masked_fill(
            #     mask, 0.0
            # )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        n_batch = v.size(0)
        attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)
        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)
        # print(x.norm(2))
        return self.linear_out(x)

class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.

    """

    def __init__(self, channels, kernel_size, activation=nn.ReLU(), bias=True):
        """Construct an ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = activation

    def forward(self, x):
        """Compute convolution module.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).

        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).

        """
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        x = self.activation(self.norm(x))

        x = self.pointwise_conv2(x)

        return x.transpose(1, 2)

class adversary_encoder_layer(torch.nn.Module):
    def __init__(self,stft_conf,MHA_conf,dropout_rate=0.0,conformer=False) -> None:
        super(adversary_encoder_layer,self).__init__()
        dim = stft_conf["fft_len"]//2+1
        if conformer:
            self.cnn_module = ConvolutionModule(dim,31)
            self.ffn_macaron = nn.Sequential(
                nn.Linear(dim,dim*4),
                nn.ReLU(),
                nn.Linear(dim*4,dim),
            )
            self.ffn_norm = LayerNorm(dim)
        else:
            self.cnn_module = None
            self.ffn_macaron = None
        self.self_attention = MultiHeadedAttention(**MHA_conf)
        self.norm1 = LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim,dim*4),
            nn.ReLU(),
            nn.Linear(dim*4,dim),
        )
        self.norm2 = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self,mag):
        # self attention
        if self.cnn_module is not None:
            residual = mag
            mag = self.cnn_module(mag)
            mag = residual + self.dropout(mag)
            residual = mag
            mag = self.ffn_norm(mag)
            mag = self.ffn_macaron(mag)
            mag = residual + self.dropout(mag)
        residual = mag
        mag = self.norm1(mag)
        mag = self.self_attention(mag,mag,mag)
        mag = self.dropout(mag) + residual
        # FFN
        residual = mag
        mag = self.norm2(mag)
        mag = self.ffn(mag)
        mag = self.dropout(mag) + residual

        return mag
class adversary_decoder_layer(torch.nn.Module):
    def __init__(self,stft_conf,MHA_conf,dropout_rate=0.0) -> None:
        super(adversary_decoder_layer,self).__init__()
        dim=stft_conf["fft_len"]//2+1
        self.dim = dim
        self.vgg_conv = nn.Sequential(
            nn.Conv2d(1,32,(3,3),(1,1),(1,1)),
            nn.ReLU(),
            nn.Conv2d(32,32,(3,3),(1,1),(1,1)),
            nn.ReLU(),
        )
        # self.vgg_conv = None
        self.out=torch.nn.Linear(dim*32, dim)
        self.norm_ffn = torch.nn.LayerNorm(dim,elementwise_affine=False)
        self.self_attention=MultiHeadedAttention(**MHA_conf)
        self.norm1 = torch.nn.LayerNorm(dim,elementwise_affine=False)
        self.cross_attention = MultiHeadedAttention(**MHA_conf)
        self.norm2 = torch.nn.LayerNorm(dim,elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim,dim*4),
            nn.ReLU(),
            nn.Linear(dim*4,dim),
        )
        self.norm3 = torch.nn.LayerNorm(dim,elementwise_affine=False)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self,mag,universal_perturbation,skip=False):
        if not skip:
            mag = self.norm1(mag)
            if self.vgg_conv is not None:
                universal_perturbation = self.vgg_conv(universal_perturbation)
                b, c, t, f = universal_perturbation.size()
                universal_perturbation = self.out(universal_perturbation.transpose(1, 2).contiguous().view(b, t, c * f))
            universal_perturbation = self.norm1(universal_perturbation)
            residual = mag
            mag = self.cross_attention(mag,universal_perturbation,universal_perturbation)
            mag = self.dropout(mag) + residual
            mag = self.norm2(mag)
            residual = mag
            mag = self.ffn(mag)
            mag = self.dropout(mag) + residual
            mag = self.norm3(mag)
        else:
            mag = self.norm1(mag)
            mag = self.norm2(mag)
            mag = self.norm3(mag)
            universal_perturbation = universal_perturbation.squeeze(1)
            universal_perturbation = universal_perturbation
            residual = mag 
            score = torch.softmax(torch.matmul(mag, universal_perturbation.transpose(-2, -1)) / math.sqrt(self.dim),dim=-1)
            attn = torch.matmul(score,universal_perturbation)
            mag = attn + residual
            mag = self.norm3(mag)
        return mag
class adversary_generator(torch.nn.Module):
    def __init__(self,noise_frame,stft_conf,MHA_conf,encoder_layers=1,decoder_layers=1,dropout_rate=0.0,universal=False,conformer=False,multi_stage=1,kmeans_selection=False) -> None:
        super(adversary_generator,self).__init__()
        dim = stft_conf["fft_len"]//2+1
        perturbation_matrix = torch.randn([1,1,noise_frame,dim])

        self.position_embedding = PositionalEncoding(dim)
        self.adversary_encoder_layer = nn.ModuleList (
            adversary_encoder_layer(stft_conf,MHA_conf,dropout_rate,conformer) for i in range (encoder_layers)
        )
        self.multi_stage = multi_stage
        if self.multi_stage > 1:
            assert multi_stage==decoder_layers

        self.universal = universal
        self.stft_extractor = adv_STFT(**stft_conf)
        self.norm = torch.nn.LayerNorm(dim,elementwise_affine=False)
    def forward(self,x,intensity=0.02,skip=False):
        # STFT
        if len(x.shape) == 3:
            x = x.squeeze(-1)
        bsz,wav_len = x.shape

        universal_x = None
        universal_mag = None
        x = F.pad(x,[0,160-wav_len%160,0,0])

        ori_mag,pha = self.stft_extractor.stft(x)
        ori_mag = ori_mag
        
        additive_mag = ori_mag
        additive_mag = additive_mag.transpose(-1,-2)
        additive_mag = self.position_embedding(additive_mag)
        for i,layer in enumerate(self.adversary_encoder_layer):
            additive_mag = layer(additive_mag)
            additive_mag = self.norm(additive_mag)
            additive_mag = additive_mag.transpose(-1,-2)
            mag = intensity*additive_mag + ori_mag
            x = self.stft_extractor.istft(mag,pha)
            x = x[:,:wav_len]
            return (x,universal_x),(ori_mag,mag,universal_mag,additive_mag)

class adv_STFT(torch.nn.Module):
    def __init__(self,fft_len=510,win_len=25,len_hop=10,sample_rate=16000) -> None:
        super(adv_STFT,self).__init__()
        self.fft_len=fft_len
        self.sample_rate=sample_rate
        self.win_len=win_len*sample_rate//1000
        self.len_hop=len_hop*sample_rate//1000
        self.window = nn.Parameter(torch.hann_window(self.win_len),requires_grad=False)
    def stft(self,wav):
        # self.window = self.window.to(wav.device)
        spec = torch.stft(wav, self.fft_len, self.len_hop, self.win_len, self.window, center=True, return_complex=False)
        # print("stft out", spec.shape)
        real = spec[:, :, :, 0]  # 实部
        imag = spec[:, :, :, 1]  # 虚部
        mags = torch.abs(torch.sqrt(torch.pow(real, 2) + torch.pow(imag, 2)))
        phase = torch.atan2(imag.data, real.data)
        return mags,phase
    def istft(self,mags,phase):
        spec=torch.concat(((torch.cos(phase)*mags).unsqueeze(-1),(torch.sin(phase)*mags).unsqueeze(-1)),dim=-1)
        wav = torch.istft(spec, self.fft_len, self.len_hop, self.win_len, self.window, True, return_complex=False)
        return wav
     
   
             