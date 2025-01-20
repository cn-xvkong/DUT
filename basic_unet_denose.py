# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Sequence, Union
import math 
import torch
import torch.nn as nn

from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import deprecated_arg, ensure_tuple_rep

__all__ = ["BasicUnet", "Basicunet", "basicunet", "BasicUNet"]

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


class TwoConv(nn.Sequential):
    """two convolutions."""

    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        dim: Optional[int] = None,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.

        .. deprecated:: 0.6.0
            ``dim`` is deprecated, use ``spatial_dims`` instead.
        """
        super().__init__()
        self.temb_proj = torch.nn.Linear(512, out_chns)

        if dim is not None:
            spatial_dims = dim
        conv_0 = Convolution(spatial_dims, in_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        conv_1 = Convolution(spatial_dims, out_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)
    
    def forward(self, x, temb):
        x = self.conv_0(x)
        x = x + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        x = self.conv_1(x)
        return x 

class Down(nn.Sequential):
    """maxpooling downsampling and two convolutions."""

    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        dim: Optional[int] = None,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.

        .. deprecated:: 0.6.0
            ``dim`` is deprecated, use ``spatial_dims`` instead.
        """
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        max_pooling = Pool["MAX", spatial_dims](kernel_size=2)
        convs = TwoConv(spatial_dims, in_chns, out_chns, act, norm, bias, dropout)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)

    def forward(self, x, temb):
        x = self.max_pooling(x)
        x = self.convs(x, temb)
        return x 

class UpCat(nn.Module):
    """upsampling, concatenation with the encoder feature map, two convolutions"""

    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        pre_conv: Optional[Union[nn.Module, str]] = "default",
        interp_mode: str = "linear",
        align_corners: Optional[bool] = True,
        halves: bool = True,
        dim: Optional[int] = None,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels to be upsampled.
            cat_chns: number of channels from the decoder.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
            pre_conv: a conv block applied before upsampling.
                Only used in the "nontrainable" or "pixelshuffle" mode.
            interp_mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``}
                Only used in the "nontrainable" mode.
            align_corners: set the align_corners parameter for upsample. Defaults to True.
                Only used in the "nontrainable" mode.
            halves: whether to halve the number of channels during upsampling.
                This parameter does not work on ``nontrainable`` mode if ``pre_conv`` is `None`.

        .. deprecated:: 0.6.0
            ``dim`` is deprecated, use ``spatial_dims`` instead.
        """
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        if upsample == "nontrainable" and pre_conv is None:
            up_chns = in_chns
        else:
            up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(
            spatial_dims,
            in_chns,
            up_chns,
            2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        self.convs = TwoConv(spatial_dims, cat_chns + up_chns, out_chns, act, norm, bias, dropout)

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor], temb):
        """

        Args:
            x: features to be upsampled.
            x_e: features from the encoder.
        """
        x_0 = self.upsample(x)

        if x_e is not None:
            # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
            dimensions = len(x.shape) - 2
            sp = [0] * (dimensions * 2)
            for i in range(dimensions):
                if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                    sp[i * 2 + 1] = 1
            x_0 = torch.nn.functional.pad(x_0, sp, "replicate")
            x = self.convs(torch.cat([x_e, x_0], dim=1), temb)  # input channels: (cat_chns + up_chns)
        else:
            x = self.convs(x_0, temb)

        return x

class conv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x


class text_classifier(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        # 自适应平均池化，将空间维度（高，宽）池化成1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 处理 num_polyps
        self.fc1 = nn.Sequential(
            nn.Linear(in_c, in_c // 8, bias=False),  # 将输入通道数减少到 in_c//8
            nn.ReLU(),
            nn.Linear(in_c // 8, out_c[0], bias=False)  # 输出 num_polyps
        )

        # 处理 polyp_sizes
        self.fc2 = nn.Sequential(
            nn.Linear(in_c, in_c // 8, bias=False),  # 将输入通道数减少到 in_c//8
            nn.ReLU(),
            nn.Linear(in_c // 8, out_c[1], bias=False)  # 输出 polyp_sizes
        )

    def forward(self, feats):
        # 对输入进行自适应平均池化，结果是 (batch_size, channels, 1, 1)
        pool = self.avg_pool(feats).view(feats.shape[0], feats.shape[1])

        # 分别传入两个全连接层
        num_polyps = self.fc1(pool)  # 预测 polyps 数量
        polyp_sizes = self.fc2(pool)  # 预测 polyp 的尺寸

        return num_polyps, polyp_sizes

class embedding_feature_fusion(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Conv2d((in_c[0]+in_c[1])*in_c[2], out_c, 1, bias=False), nn.ReLU(),
            nn.Conv2d(out_c, out_c, 1, bias=False), nn.ReLU()
        )

    def forward(self, num_polyps, polyp_sizes, label):
        num_polyps_prob = torch.softmax(num_polyps, axis=1)
        polyp_sizes_prob = torch.softmax(polyp_sizes, axis=1)
        prob = torch.cat([num_polyps_prob, polyp_sizes_prob], axis=1)
        prob = prob.view(prob.shape[0], prob.shape[1], 1)
        x = label * prob
        x = x.view(x.shape[0], -1, 1, 1)
        x = self.fc(x)
        x = x.view(x.shape[0], -1)
        return x

class text_attention(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.c1 = nn.Sequential(
            nn.Conv2d(in_c[1], in_c[0], kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_c[0], in_c[0], kernel_size=1, padding=0, bias=False)
        )

    def forward(self, feats, label):
        """ Channel Attention """
        b, c = label.shape
        label = label.reshape(b, c, 1, 1)
        ch_attn = self.c1(label)
        ch_map = torch.sigmoid(ch_attn)
        feats = feats * ch_map

        ch_attn = ch_attn.reshape(ch_attn.shape[0], ch_attn.shape[1])
        return ch_attn, feats


class multiscale_feature_aggregation(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.up_2x2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_4x4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

        self.c11 = conv2d(in_c[0], out_c, kernel_size=1, padding=0)
        self.c12 = conv2d(in_c[1], out_c, kernel_size=1, padding=0)
        self.c13 = conv2d(in_c[2], out_c, kernel_size=1, padding=0)
        self.c14 = conv2d(in_c[3], out_c, kernel_size=1, padding=0)
        self.c15 = conv2d(out_c * 4, out_c, kernel_size=1, padding=0)

        self.c2 = conv2d(out_c, out_c, act=False)
        self.c3 = conv2d(out_c, out_c, act=False)

    def forward(self, x1, x2, x3, x4):
        x1 = self.up_4x4(x1)
        x2 = self.up_2x2(x2)

        x1 = self.c11(self.up_2x2(x1))
        x2 = self.c12(self.up_2x2(x2))
        x3 = self.c13(self.up_2x2(x3))
        x4 = self.c14(x4)
        x = torch.cat([x1, x2, x3, x4], axis=1)
        x = self.c15(x)

        s1 = x
        x = self.c2(x)
        x = self.relu(x+s1)

        s2 = x
        x = self.c3(x)
        x = self.relu(x+s2+s1)

        return x

class channel_attention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x0 * self.sigmoid(out)

class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x0 * self.sigmoid(x)

class dilated_conv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.c1 = nn.Sequential(conv2d(in_c, out_c, kernel_size=1, padding=0), channel_attention(out_c))
        self.c2 = nn.Sequential(conv2d(in_c, out_c, kernel_size=(3, 3), padding=6, dilation=6), channel_attention(out_c))
        self.c3 = nn.Sequential(conv2d(in_c, out_c, kernel_size=(3, 3), padding=12, dilation=12), channel_attention(out_c))
        self.c4 = nn.Sequential(conv2d(in_c, out_c, kernel_size=(3, 3), padding=18, dilation=18), channel_attention(out_c))
        self.c5 = conv2d(out_c*4, out_c, kernel_size=3, padding=1, act=False)
        self.c6 = conv2d(in_c, out_c, kernel_size=1, padding=0, act=False)
        self.sa = spatial_attention()

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)
        xc = torch.cat([x1, x2, x3, x4], axis=1)
        xc = self.c5(xc)
        xs = self.c6(x)
        x = self.relu(xc+xs)
        x = self.sa(x)
        return x

class BasicUNetDe(nn.Module):
    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        dimensions: Optional[int] = None,
    ):
        """
        A UNet implementation with 1D/2D/3D supports.

        Based on:

            Falk et al. "U-Net – Deep Learning for Cell Counting, Detection, and
            Morphometry". Nature Methods 16, 67–70 (2019), DOI:
            http://dx.doi.org/10.1038/s41592-018-0261-2

        Args:
            spatial_dims: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
            in_channels: number of input channels. Defaults to 1.
            out_channels: number of output channels. Defaults to 2.
            features: six integers as numbers of features.
                Defaults to ``(32, 32, 64, 128, 256, 32)``,

                - the first five values correspond to the five-level encoder feature sizes.
                - the last value corresponds to the feature size after the last upsampling.

            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            bias: whether to have a bias term in convolution blocks. Defaults to True.
                According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
                if a conv layer is directly followed by a batch norm layer, bias should be False.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.

        .. deprecated:: 0.6.0
            ``dimensions`` is deprecated, use ``spatial_dims`` instead.

        Examples::

            # for spatial 2D
            >>> net = BasicUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128))

            # for spatial 2D, with group norm
            >>> net = BasicUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128), norm=("group", {"num_groups": 4}))

            # for spatial 3D
            >>> net = BasicUNet(spatial_dims=3, features=(32, 32, 64, 128, 256, 32))

        See Also

            - :py:class:`monai.networks.nets.DynUNet`
            - :py:class:`monai.networks.nets.UNet`

        """
        super().__init__()
        if dimensions is not None:
            spatial_dims = dimensions

        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")
        
        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(128,
                            512),
            torch.nn.Linear(512,
                            512),
        ])

        self.conv_0 = TwoConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)

        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[0], act, norm, bias, dropout, upsample, halves=False)

        self.text_classifier = text_classifier(1024, [2, 3])
        self.label_fc = embedding_feature_fusion([2, 3, 300], 128)

        self.s1 = dilated_conv(128, 128)
        self.s2 = dilated_conv(256, 128)
        self.s3 = dilated_conv(512, 128)
        self.s4 = dilated_conv(1024, 128)

        self.a1 = text_attention([128, 128])
        self.a2 = text_attention([128, 128])
        self.a3 = text_attention([128, 128])
        self.a4 = text_attention([128, 128])

        self.MSFA = multiscale_feature_aggregation([128, 128, 128, 128], 128)

        self.final_conv = Conv["conv", spatial_dims](128, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t, embeddings=None, image=None, text=None):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N])``.
        """
        temb = get_timestep_embedding(t, 128)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        if image is not None :
            x = torch.cat([image, x], dim=1)
            
        x0 = self.conv_0(x, temb)
        if embeddings is not None:
            x0 += embeddings[0]

        x1 = self.down_1(x0, temb)
        if embeddings is not None:
            x1 += embeddings[1]

        x2 = self.down_2(x1, temb)
        if embeddings is not None:
            x2 += embeddings[2]

        x3 = self.down_3(x2, temb)
        if embeddings is not None:
            x3 += embeddings[3]

        x4 = self.down_4(x3, temb)
        if embeddings is not None:
            x4 += embeddings[4]

        num_polyps, polyp_sizes = self.text_classifier(x4)
        f0 = self.label_fc(num_polyps, polyp_sizes, text)

        """ Dilated Conv """
        s1 = self.s1(x1)
        s2 = self.s2(x2)
        s3 = self.s3(x3)
        s4 = self.s4(x4)

        u4 = self.upcat_4(x4, x3, temb)
        f1, a1 = self.a1(u4, f0)
        f = f0 + f1

        u3 = self.upcat_3(u4, x2, temb)
        f2, a2 = self.a2(u3, f)
        f = f0 + f1 + f2

        u2 = self.upcat_2(u3, x1, temb)
        f3, a3 = self.a3(u2, f)
        f = f0 + f1 + f2 + f3

        u1 = self.upcat_1(u2, x0, temb)
        f4, a4 = self.a4(u1, f)

        msfa = self.MSFA(a1, a2, a3, a4)

        logits = self.final_conv(msfa)
        return logits



