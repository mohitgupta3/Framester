import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        nn.PReLU(out_planes),
    )


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=True,
        ),
        nn.PReLU(out_planes),
    )


class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Unet(nn.Module):
    def __init__(self, c, out=3):
        super(Unet, self).__init__()
        self.down0 = Conv2(17 + c, 2 * c)
        self.down1 = Conv2(4 * c, 4 * c)
        self.down2 = Conv2(8 * c, 8 * c)
        self.down3 = Conv2(16 * c, 16 * c)
        self.up0 = deconv(32 * c, 8 * c)
        self.up1 = deconv(16 * c, 4 * c)
        self.up2 = deconv(8 * c, 2 * c)
        self.up3 = deconv(4 * c, c)
        self.conv = nn.Conv2d(c, out, 3, 1, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, img0, img1, warped_img0, warped_img1, mask, flow, c0, c1):
        s0 = self.down0(
            torch.cat(
                (img0, img1, warped_img0, warped_img1, mask, flow, c0[0], c1[0]), 1
            )
        )
        s1 = self.down1(torch.cat((s0, c0[1], c1[1]), 1))
        s2 = self.down2(torch.cat((s1, c0[2], c1[2]), 1))
        s3 = self.down3(torch.cat((s2, c0[3], c1[3]), 1))
        x = self.up0(torch.cat((s3, c0[4], c1[4]), 1))
        x = self.up1(torch.cat((x, s2), 1))
        x = self.up2(torch.cat((x, s1), 1))
        x = self.up3(torch.cat((x, s0), 1))
        x = self.conv(x)
        return torch.sigmoid(x)


class Head(nn.Module):
    def __init__(self, in_planes, scale, c, in_else=17):
        super(Head, self).__init__()
        self.upsample = nn.Sequential(nn.PixelShuffle(2), nn.PixelShuffle(2))
        self.scale = scale
        self.conv = nn.Sequential(
            conv(in_planes * 2 // (4 * 4) + in_else, c),
            conv(c, c),
            conv(c, 5),
        )

    def forward(self, motion_feature, x, flow):  # /16 /8 /4
        motion_feature = self.upsample(motion_feature)  # /4 /2 /1
        if self.scale != 4:
            x = F.interpolate(
                x, scale_factor=4.0 / self.scale, mode="bilinear", align_corners=False
            )
        if flow != None:
            if self.scale != 4:
                flow = (
                    F.interpolate(
                        flow,
                        scale_factor=4.0 / self.scale,
                        mode="bilinear",
                        align_corners=False,
                    )
                    * 4.0
                    / self.scale
                )
            x = torch.cat((x, flow), 1)
        x = self.conv(torch.cat([motion_feature, x], 1))
        if self.scale != 4:
            x = F.interpolate(
                x, scale_factor=self.scale // 4, mode="bilinear", align_corners=False
            )
            flow = x[:, :4] * (self.scale // 4)
        else:
            flow = x[:, :4]
        mask = x[:, 4:5]
        return flow, mask


class MultiScaleFlow(nn.Module):
    def __init__(self, backbone, **kargs):
        super(MultiScaleFlow, self).__init__()
        self.flow_num_stage = len(kargs["hidden_dims"])
        self.feature_bone = backbone
        self.block = nn.ModuleList(
            [
                Head(
                    kargs["motion_dims"][-1 - i] * kargs["depths"][-1 - i]
                    + kargs["embed_dims"][-1 - i],
                    kargs["scales"][-1 - i],
                    kargs["hidden_dims"][-1 - i],
                    6 if i == 0 else 17,
                )
                for i in range(self.flow_num_stage)
            ]
        )
        self.unet = Unet(kargs["c"] * 2)

        self.backwarp_tenGrid = {}

    def warp(self, tenInput, tenFlow):
        k = (str(tenFlow.device), str(tenFlow.size()))
        if k not in self.backwarp_tenGrid:
            tenHorizontal = (
                torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device)
                .view(1, 1, 1, tenFlow.shape[3])
                .expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
            )
            tenVertical = (
                torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device)
                .view(1, 1, tenFlow.shape[2], 1)
                .expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
            )
            self.backwarp_tenGrid[k] = torch.cat([tenHorizontal, tenVertical], 1).to(
                device
            )

        tenFlow = torch.cat(
            [
                tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0),
            ],
            1,
        )

        g = (self.backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
        return torch.nn.functional.grid_sample(
            input=tenInput,
            grid=g,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

    def warp_features(self, xs, flow):
        y0 = []
        y1 = []
        B = xs[0].size(0) // 2
        for x in xs:
            y0.append(self.warp(x[:B], flow[:, 0:2]))
            y1.append(self.warp(x[B:], flow[:, 2:4]))
            flow = (
                F.interpolate(
                    flow,
                    scale_factor=0.5,
                    mode="bilinear",
                    align_corners=False,
                    recompute_scale_factor=False,
                )
                * 0.5
            )
        return y0, y1

    def calculate_flow(self, imgs, timestep, af=None, mf=None):
        img0, img1 = imgs[:, :3], imgs[:, 3:6]
        B = img0.size(0)
        flow, mask = None, None
        # appearence_features & motion_features
        if (af is None) or (mf is None):
            af, mf = self.feature_bone(img0, img1)
        for i in range(self.flow_num_stage):
            t = torch.full(mf[-1 - i][:B].shape, timestep, dtype=torch.float).cuda()
            if flow != None:
                warped_img0 = self.warp(img0, flow[:, :2])
                warped_img1 = self.warp(img1, flow[:, 2:4])
                flow_, mask_ = self.block[i](
                    torch.cat(
                        [
                            t * mf[-1 - i][:B],
                            (1 - t) * mf[-1 - i][B:],
                            af[-1 - i][:B],
                            af[-1 - i][B:],
                        ],
                        1,
                    ),
                    torch.cat((img0, img1, warped_img0, warped_img1, mask), 1),
                    flow,
                )
                flow = flow + flow_
                mask = mask + mask_
            else:
                flow, mask = self.block[i](
                    torch.cat(
                        [
                            t * mf[-1 - i][:B],
                            (1 - t) * mf[-1 - i][B:],
                            af[-1 - i][:B],
                            af[-1 - i][B:],
                        ],
                        1,
                    ),
                    torch.cat((img0, img1), 1),
                    None,
                )

        return flow, mask

    def coraseWarp_and_Refine(self, imgs, af, flow, mask):
        img0, img1 = imgs[:, :3], imgs[:, 3:6]
        warped_img0 = self.warp(img0, flow[:, :2])
        warped_img1 = self.warp(img1, flow[:, 2:4])
        c0, c1 = self.warp_features(af, flow)
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[:, :3] * 2 - 1
        mask_ = torch.sigmoid(mask)
        merged = warped_img0 * mask_ + warped_img1 * (1 - mask_)
        pred = torch.clamp(merged + res, 0, 1)
        return pred

    # Actually consist of 'calculate_flow' and 'coraseWarp_and_Refine'
    def forward(self, x, timestep=0.5):
        img0, img1 = x[:, :3], x[:, 3:6]
        B = x.size(0)
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        # appearence_features & motion_features
        af, mf = self.feature_bone(img0, img1)
        for i in range(self.flow_num_stage):
            t = torch.full(mf[-1 - i][:B].shape, timestep, dtype=torch.float).cuda()
            if flow != None:
                flow_d, mask_d = self.block[i](
                    torch.cat(
                        [
                            t * mf[-1 - i][:B],
                            (1 - timestep) * mf[-1 - i][B:],
                            af[-1 - i][:B],
                            af[-1 - i][B:],
                        ],
                        1,
                    ),
                    torch.cat((img0, img1, warped_img0, warped_img1, mask), 1),
                    flow,
                )
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = self.block[i](
                    torch.cat(
                        [
                            t * mf[-1 - i][:B],
                            (1 - t) * mf[-1 - i][B:],
                            af[-1 - i][:B],
                            af[-1 - i][B:],
                        ],
                        1,
                    ),
                    torch.cat((img0, img1), 1),
                    None,
                )
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = self.warp(img0, flow[:, :2])
            warped_img1 = self.warp(img1, flow[:, 2:4])
            merged.append(warped_img0 * mask_list[i] + warped_img1 * (1 - mask_list[i]))

        c0, c1 = self.warp_features(af, flow)
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[:, :3] * 2 - 1
        pred = torch.clamp(merged[-1] + res, 0, 1)
        return flow_list, mask_list, merged, pred
