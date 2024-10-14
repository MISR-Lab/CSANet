import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import default_init_weights, make_layer

import math
import numpy as np



class CSA(nn.Module):
    '''Channel Shift Attention''' # ,
    def __init__(self, dim=64, ksize=3,lambda_dispersion=3, shift_cpg=4, stab_cpg=3,exp_ord=0, reduction=16):
        super(CSA, self).__init__()
        self.ksize = ksize
        self.dim = dim


        if ksize==5:

            if exp_ord == 0:
                CS_kernel_pos = [7, 13, 17, 11, 0, 4, 24, 20]
            elif exp_ord == 1:
                CS_kernel_pos = [2, 14, 22, 10, 6, 8, 18, 16]
            else:
                CS_kernel_pos = [2, 14, 22, 10, 0, 4, 24, 20]

        else:
            CS_kernel_pos = [1, 5, 7, 3, 0, 2, 8, 6]



        weights_cw = torch.zeros(dim, 1, ksize, ksize)

        stab_parts = 2**lambda_dispersion # == shift_parts

        channel_of_sub_groups = (len(CS_kernel_pos)/stab_parts)*shift_cpg + stab_cpg  # 需要考虑分散系数。

        for idx in range(dim):

            groups_id = (idx // channel_of_sub_groups) % stab_parts
            channel_id = (idx % channel_of_sub_groups)
            is_stability = channel_id // ((len(CS_kernel_pos)/stab_parts)*shift_cpg) # =0不是，=1则是：


            if is_stability:
                if self.ksize == 5:
                    ker_pos = 12
                else:
                    ker_pos = 4  # 即保持shift_kerner中心位置为1
            else:
                pos = (((channel_of_sub_groups-stab_cpg )*groups_id) + channel_id ) // shift_cpg
                ker_pos = CS_kernel_pos[int(pos)]

            weight_s = torch.zeros(ksize * ksize)
            weight_s[ker_pos] = 1
            weight_s_reshape = torch.reshape(weight_s, (1, 1, ksize, ksize))
            weights_cw[idx,:, :, :] = weight_s_reshape
        self.weight_cw = weights_cw


        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=False),
            # nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

    def forward(self, x):

        atten = self.conv_du(self.avg_pool(x))

        weight_cw = self.weight_cw.to(x.device)
        out = F.conv2d(x, weight_cw, padding=(self.ksize - 1) // 2, groups=self.dim)

        res = out
        out = torch.mul(out, atten)


        return out+ res

class CSAB(nn.Module):
    '''Channel Shift Attention Block'''
    def __init__(self, nfeats=64, ksize=3, lamb_dis=1,shift_cpg=4, stab_cpg=3,exp_ord=0, reduction=16):
        super(CSAB, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(nfeats, nfeats, 1),
            CSA(nfeats, ksize, lamb_dis, shift_cpg,stab_cpg,exp_ord,reduction),
            nn.ReLU(True),
            nn.Conv2d(nfeats, nfeats, 1),
        )

    def forward(self, x):
        res = x
        x = self.body(x)
        return res + x
    
@ARCH_REGISTRY.register()
class CSANet(nn.Module):
    '''  Channel Shift Attention Network    '''

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, ksize=3, upscale=2, groups=3, shift_cpg=4,
                 stab_cpg=2, exp_ord=0, reduction=16):
        super(CSANet, self).__init__()

        self.scale = upscale
        self.head = nn.Sequential(
            nn.Conv2d(num_in_ch, num_feat, 1),
            CSA(num_feat, ksize, groups, shift_cpg, stab_cpg, exp_ord, reduction),
            nn.Conv2d(num_feat, num_feat, 1),

        )

        self.body = make_layer(CSAB, num_block, nfeats=num_feat, ksize=ksize, lamb_dis=groups, shift_cpg=shift_cpg,
                               stab_cpg=stab_cpg, exp_ord=exp_ord,reduction=reduction)

        if self.scale in [2, 3]:
            self.tail = nn.Sequential(

                nn.Conv2d(num_feat, num_feat, 1),
                CSA(num_feat, ksize, groups, shift_cpg, stab_cpg, exp_ord, reduction),
                # nn.ReLU(True),

                nn.Conv2d(num_feat, num_feat * upscale ** 2, 1),
                nn.PixelShuffle(upscale),

                nn.Conv2d(num_feat, num_feat, 1),
                CSA(num_feat, ksize, groups, shift_cpg, stab_cpg, exp_ord, reduction),
                # nn.ReLU(True),
                nn.Conv2d(num_feat, num_feat, 1),



                nn.Conv2d(num_feat, num_out_ch, 1),
            )

        elif self.scale == 4:
            self.tail = nn.Sequential(

                nn.Conv2d(num_feat, num_feat, 1),
                CSA(num_feat, ksize, groups, shift_cpg, stab_cpg, exp_ord, reduction),
                # nn.ReLU(True),

                nn.Conv2d(num_feat, num_feat * 2 ** 2, 1),
                nn.PixelShuffle(2),

                nn.Conv2d(num_feat, num_feat, 1),
                CSA(num_feat, ksize, groups, shift_cpg, stab_cpg, exp_ord, reduction),
                # nn.ReLU(True),

                nn.Conv2d(num_feat, num_feat * 2 ** 2, 1),
                nn.PixelShuffle(2),

                nn.Conv2d(num_feat, num_feat, 1),
                CSA(num_feat, ksize, groups, shift_cpg, stab_cpg, exp_ord, reduction),
                # nn.ReLU(True),
                nn.Conv2d(num_feat, num_feat, 1),

                nn.Conv2d(num_feat, num_out_ch, 1),
            )

    def forward(self, x):
        res = x
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        res_up = F.interpolate(res, scale_factor=self.scale, mode='bicubic', align_corners=False)

        return x + res_up