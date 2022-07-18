# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda import amp

from compressai.entropy_models import GaussianConditional
from compressai.layers import QReLU

from mmcv.ops import ModulatedDeformConv2d as DCN

from ..google import CompressionModel, get_scale_table
from ..utils import (
    conv,
    deconv,
    gaussian_blur,
    gaussian_kernel2d,
    meshgrid2d,
    quantize_ste,
    update_registered_buffers,
)


class FVC_base(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        
        out_channel_F = 64
        out_channel_O = 64
        out_channel_M = 128
        deform_ks = 3
        deform_groups = 8
        
        self.out_channel_F = out_channel_F
        self.out_channel_O = out_channel_O
        self.out_channel_M = out_channel_M
        self.deform_ks = deform_ks
        self.deform_groups = deform_groups
            
        class ResBlock(nn.Module):
            def __init__(self, inputchannel, outputchannel, kernel_size=3, stride=1):
                super(ResBlock, self).__init__()
                self.relu1 = nn.ReLU()
                self.conv1 = nn.Conv2d(inputchannel, outputchannel,
                                    kernel_size, stride, padding=kernel_size//2)
                torch.nn.init.xavier_uniform_(self.conv1.weight.data)
                torch.nn.init.constant_(self.conv1.bias.data, 0.0)
                self.relu2 = nn.ReLU()
                self.conv2 = nn.Conv2d(outputchannel, outputchannel,
                                    kernel_size, stride, padding=kernel_size//2)
                torch.nn.init.xavier_uniform_(self.conv2.weight.data)
                torch.nn.init.constant_(self.conv2.bias.data, 0.0)

            def forward(self, x):
                x_1 = self.relu1(x)
                firstlayer = self.conv1(x_1)
                firstlayer = self.relu2(firstlayer)
                seclayer = self.conv2(firstlayer)
                return x + seclayer
         
        self.feature_extract = nn.Sequential(
            conv(3, out_channel_F, kernel_size=5, stride=2),
            ResBlock(out_channel_F, out_channel_F),
            ResBlock(out_channel_F, out_channel_F),
            ResBlock(out_channel_F, out_channel_F),
        )
        
        self.frame_reconstruct = nn.Sequential(
            ResBlock(out_channel_F, out_channel_F),
            ResBlock(out_channel_F, out_channel_F),
            ResBlock(out_channel_F, out_channel_F),
            deconv(out_channel_F, 3, kernel_size=5, stride=2),
        )
        
        self.motion_estimate = nn.Sequential(
            conv(out_channel_F * 2, out_channel_O, kernel_size=3, stride=1),
            conv(out_channel_O, out_channel_O, kernel_size=3, stride=1),
        )
        
        self.motion_compensate = nn.Sequential(
            conv(out_channel_F * 2, out_channel_F, kernel_size=3, stride=1),
            conv(out_channel_F, out_channel_F, kernel_size=3, stride=1),
        )
        
        self.offset_mask_conv = nn.Conv2d(
            out_channel_O,
            deform_groups * 3 * deform_ks * deform_ks,
            kernel_size=deform_ks,
            stride=1,
            padding=1,
        )
        
        self.deform_conv = DCN(
            out_channel_F, out_channel_F, kernel_size=(deform_ks, deform_ks), padding=deform_ks//2, deform_groups=deform_groups
            )
        
        class EncoderWithResblock(nn.Sequential):
            def __init__(
                self, in_planes: int, out_planes: int
            ):
                super().__init__(
                    conv(in_planes, out_planes, kernel_size=3, stride=2),
                    ResBlock(out_planes, out_planes),
                    ResBlock(out_planes, out_planes),
                    ResBlock(out_planes, out_planes),
                    conv(out_planes, out_planes, kernel_size=3, stride=2),
                    ResBlock(out_planes, out_planes),
                    ResBlock(out_planes, out_planes),
                    ResBlock(out_planes, out_planes),
                    conv(out_planes, out_planes, kernel_size=3, stride=2),
                    ResBlock(out_planes, out_planes),
                    ResBlock(out_planes, out_planes),
                    ResBlock(out_planes, out_planes),
                )
                
        class DecoderWithResblock(nn.Sequential):
            def __init__(
                self, in_planes: int, out_planes: int
            ):
                super().__init__(
                    ResBlock(in_planes, in_planes),
                    ResBlock(in_planes, in_planes),
                    ResBlock(in_planes, in_planes),
                    deconv(in_planes, in_planes, kernel_size=3, stride=2),
                    ResBlock(in_planes, in_planes),
                    ResBlock(in_planes, in_planes),
                    ResBlock(in_planes, in_planes),
                    deconv(in_planes, in_planes, kernel_size=3, stride=2),
                    ResBlock(in_planes, in_planes),
                    ResBlock(in_planes, in_planes),
                    ResBlock(in_planes, in_planes),
                    deconv(in_planes, out_planes, kernel_size=3, stride=2),
                )
                
        class Encoder(nn.Sequential):
            def __init__(
                self, in_planes: int, mid_planes: int = 128, out_planes: int = 192
            ):
                super().__init__(
                    conv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    conv(mid_planes, out_planes, kernel_size=5, stride=2),
                )

        class Decoder(nn.Sequential):
            def __init__(
                self, out_planes: int, in_planes: int = 192, mid_planes: int = 128
            ):
                super().__init__(
                    deconv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    deconv(mid_planes, out_planes, kernel_size=5, stride=2),
                )

        class HyperEncoder(nn.Sequential):
            def __init__(
                self, in_planes: int = 192, mid_planes: int = 192, out_planes: int = 192
            ):
                super().__init__(
                    conv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                )

        class HyperDecoder(nn.Sequential):
            def __init__(
                self, in_planes: int = 192, mid_planes: int = 192, out_planes: int = 192
            ):
                super().__init__(
                    deconv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    deconv(mid_planes, out_planes, kernel_size=5, stride=2),
                )

        class HyperDecoderWithQReLU(nn.Module):
            def __init__(
                self, in_planes: int = 192, mid_planes: int = 192, out_planes: int = 192
            ):
                super().__init__()

                def qrelu(input, bit_depth=8, beta=100):
                    return QReLU.apply(input, bit_depth, beta)

                self.deconv1 = deconv(in_planes, mid_planes, kernel_size=5, stride=2)
                self.qrelu1 = qrelu
                self.deconv2 = deconv(mid_planes, mid_planes, kernel_size=5, stride=2)
                self.qrelu2 = qrelu
                self.deconv3 = deconv(mid_planes, out_planes, kernel_size=5, stride=2)
                self.qrelu3 = qrelu

            def forward(self, x):
                x = self.qrelu1(self.deconv1(x))
                x = self.qrelu2(self.deconv2(x))
                x = self.qrelu3(self.deconv3(x))

                return x

        class Hyperprior(CompressionModel):
            def __init__(self, planes: int = 192, mid_planes: int = 192):
                super().__init__(entropy_bottleneck_channels=mid_planes)
                self.hyper_encoder = HyperEncoder(planes, mid_planes, planes)
                self.hyper_decoder_mean = HyperDecoder(planes, mid_planes, planes)
                self.hyper_decoder_scale = HyperDecoderWithQReLU(
                    planes, mid_planes, planes
                )
                self.gaussian_conditional = GaussianConditional(None)

            def forward(self, y):
                z = self.hyper_encoder(y)
                z_hat, z_likelihoods = self.entropy_bottleneck(z)

                scales = self.hyper_decoder_scale(z_hat)
                means = self.hyper_decoder_mean(z_hat)
                _, y_likelihoods = self.gaussian_conditional(y, scales, means)
                y_hat = quantize_ste(y - means) + means
                return y_hat, {"y": y_likelihoods, "z": z_likelihoods}

            def compress(self, y):
                z = self.hyper_encoder(y)

                z_string = self.entropy_bottleneck.compress(z)
                z_hat = self.entropy_bottleneck.decompress(z_string, z.size()[-2:])

                scales = self.hyper_decoder_scale(z_hat)
                means = self.hyper_decoder_mean(z_hat)

                indexes = self.gaussian_conditional.build_indexes(scales)
                y_string = self.gaussian_conditional.compress(y, indexes, means)
                y_hat = self.gaussian_conditional.quantize(y, "dequantize", means)

                return y_hat, {"strings": [y_string, z_string], "shape": z.size()[-2:]}

            def decompress(self, strings, shape):
                assert isinstance(strings, list) and len(strings) == 2
                z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

                scales = self.hyper_decoder_scale(z_hat)
                means = self.hyper_decoder_mean(z_hat)
                indexes = self.gaussian_conditional.build_indexes(scales)
                y_hat = self.gaussian_conditional.decompress(
                    strings[0], indexes, z_hat.dtype, means
                )

                return y_hat
            
        self.img_encoder = Encoder(3)
        self.img_decoder = Decoder(3)
        self.img_hyperprior = Hyperprior()

        self.motion_encoder = EncoderWithResblock(in_planes=out_channel_O, out_planes=out_channel_M)
        self.motion_decoder = DecoderWithResblock(in_planes=out_channel_M, out_planes=out_channel_O)
        self.motion_hyperprior = Hyperprior(planes=out_channel_M, mid_planes=out_channel_M)
        
        self.res_encoder = EncoderWithResblock(in_planes=out_channel_F, out_planes=128)
        self.res_decoder = DecoderWithResblock(in_planes=128, out_planes=out_channel_F)
        self.res_hyperprior = Hyperprior(planes=128, mid_planes=128)

    def forward(self, frames):
        if not isinstance(frames, List):
            raise RuntimeError(f"Invalid number of frames: {len(frames)}.")

        reconstructions = []
        frames_likelihoods = []

        x_hat, likelihoods = self.forward_keyframe(frames[0])
        reconstructions.append(x_hat)
        frames_likelihoods.append(likelihoods)
        x_ref = x_hat.detach()  # stop gradient flow (cf: google2020 paper)

        for i in range(1, len(frames)):
            x = frames[i]
            x_ref, likelihoods = self.forward_inter(x, x_ref)
            reconstructions.append(x_ref)
            frames_likelihoods.append(likelihoods)

        return {
            "x_hat": reconstructions,
            "likelihoods": frames_likelihoods,
        }

    def forward_keyframe(self, x):
        y = self.img_encoder(x)
        y_hat, likelihoods = self.img_hyperprior(y)
        x_hat = self.img_decoder(y_hat)
        return x_hat, {"keyframe": likelihoods}

    def encode_keyframe(self, x):
        y = self.img_encoder(x)
        y_hat, out_keyframe = self.img_hyperprior.compress(y)
        x_hat = self.img_decoder(y_hat)

        return x_hat, out_keyframe

    def decode_keyframe(self, strings, shape):
        y_hat = self.img_hyperprior.decompress(strings, shape)
        x_hat = self.img_decoder(y_hat)

        return x_hat
    
    def forward_inter(self, x_cur, x_ref):
        # feature extraction
        f_cur = self.feature_extract(x_cur)
        f_ref = self.feature_extract(x_ref)
        f = torch.cat((f_cur, f_ref), dim=1)
        
        # motion estimation
        offset = self.motion_estimate(f)
        # encode motion info
        offset = self.motion_encoder(offset)
        offset_hat, motion_likelihoods = self.motion_hyperprior(offset)
        # decode motion info
        offset_info = self.motion_decoder(offset_hat)
        
        # motion compensation
        deformed_f_ref = self.dcn_warp(offset_info, f_ref)
        f_mc = torch.cat((deformed_f_ref, f_ref), dim=1)
        f_mc = self.motion_compensate(f_mc)
        f_bar = deformed_f_ref + f_mc
        
        # feature space residual
        res = f_cur - f_bar
        encoded_res = self.res_encoder(res)
        f_res_hat, f_res_likelihoods = self.res_hyperprior(encoded_res)
        f_res = self.res_decoder(f_res_hat)
        
        # frame construction
        f_combine = f_res + f_bar
        x_rec = self.frame_reconstruct(f_combine)
        
        return x_rec, {"motion": motion_likelihoods, "residual": f_res_likelihoods}

    def dcn_warp(self, offset_info, f_ref):
        offset_and_mask = self.offset_mask_conv(offset_info)
        o1, o2, mask = torch.chunk(offset_and_mask, 3, dim=1)
        offset_map = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        
        return self.deform_conv(f_ref, offset_map, mask)
    
    def encode_inter(self, x_cur, x_ref):
        # feature extraction
        f_cur = self.feature_extract(x_cur)
        f_ref = self.feature_extract(x_ref)
        f = torch.cat((f_cur, f_ref), dim=1)
        
        # motion estimation
        offset = self.motion_estimate(f)
        # encode motion info
        offset = self.motion_encoder(offset)
        offset_hat, out_motion = self.motion_hyperprior.compress(offset)
        # decode motion info
        offset_info = self.motion_decoder(offset_hat)
        
        # motion compensation
        deformed_f_ref = self.dcn_warp(offset_info, f_ref)
        f_mc = torch.cat((deformed_f_ref, f_ref), dim=1)
        f_mc = self.motion_compensate(f_mc)
        f_bar = deformed_f_ref + f_mc
        
        # feature space residual
        res = f_cur - f_bar
        encoded_res = self.res_encoder(res)
        f_res_hat, out_res = self.res_hyperprior.compress(encoded_res)
        f_res = self.res_decoder(f_res_hat)
        
        # frame construction
        f_combine = f_res + f_bar
        x_rec = self.frame_reconstruct(f_combine)
        
        return x_rec, {
            "strings": {
                "motion": out_motion["strings"],
                "residual": out_res["strings"],
            },
            "shape": {"motion": out_motion["shape"], "residual": out_res["shape"]},
        }

    def decode_inter(self, x_ref, strings, shapes):
        # motion
        key = "motion"
        offset_hat = self.motion_hyperprior.decompress(strings[key], shapes[key])
        offset_info = self.motion_decoder(offset_hat)
        
        f_ref = self.feature_extract(x_ref)
        deformed_f_ref = self.dcn_warp(offset_info, f_ref)
        f_mc = torch.cat((deformed_f_ref, f_ref), dim=1)
        f_mc = self.motion_compensate(f_mc)
        f_bar = deformed_f_ref + f_mc

        # residual
        key = "residual"
        f_res_hat = self.res_hyperprior.decompress(strings[key], shapes[key])
        f_res = self.res_decoder(f_res_hat)
        
        f_combine = f_res + f_bar
        x_rec = self.frame_reconstruct(f_combine)

        return x_rec

    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""

        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel):
                aux_loss_list.append(m.aux_loss())

        return aux_loss_list

    def compress(self, frames):
        if not isinstance(frames, List):
            raise RuntimeError(f"Invalid number of frames: {len(frames)}.")

        frame_strings = []
        shape_infos = []

        x_ref, out_keyframe = self.encode_keyframe(frames[0])

        frame_strings.append(out_keyframe["strings"])
        shape_infos.append(out_keyframe["shape"])

        for i in range(1, len(frames)):
            x = frames[i]
            x_ref, out_interframe = self.encode_inter(x, x_ref)

            frame_strings.append(out_interframe["strings"])
            shape_infos.append(out_interframe["shape"])

        return frame_strings, shape_infos

    def decompress(self, strings, shapes):

        if not isinstance(strings, List) or not isinstance(shapes, List):
            raise RuntimeError(f"Invalid number of frames: {len(strings)}.")

        assert len(strings) == len(
            shapes
        ), f"Number of information should match {len(strings)} != {len(shapes)}."

        dec_frames = []

        x_ref = self.decode_keyframe(strings[0], shapes[0])
        dec_frames.append(x_ref)

        for i in range(1, len(strings)):
            string = strings[i]
            shape = shapes[i]
            x_ref = self.decode_inter(x_ref, string, shape)
            dec_frames.append(x_ref)

        return dec_frames

    def load_state_dict(self, state_dict):

        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.img_hyperprior.gaussian_conditional,
            "img_hyperprior.gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.img_hyperprior.entropy_bottleneck,
            "img_hyperprior.entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )

        update_registered_buffers(
            self.res_hyperprior.gaussian_conditional,
            "res_hyperprior.gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.res_hyperprior.entropy_bottleneck,
            "res_hyperprior.entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )

        update_registered_buffers(
            self.motion_hyperprior.gaussian_conditional,
            "motion_hyperprior.gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.motion_hyperprior.entropy_bottleneck,
            "motion_hyperprior.entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )

        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict)
        return net

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()

        updated = self.img_hyperprior.gaussian_conditional.update_scale_table(
            scale_table, force=force
        )

        updated |= self.img_hyperprior.entropy_bottleneck.update(force=force)

        updated |= self.res_hyperprior.gaussian_conditional.update_scale_table(
            scale_table, force=force
        )
        updated |= self.res_hyperprior.entropy_bottleneck.update(force=force)

        updated |= self.motion_hyperprior.gaussian_conditional.update_scale_table(
            scale_table, force=force
        )
        updated |= self.motion_hyperprior.entropy_bottleneck.update(force=force)

        return updated
