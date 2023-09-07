import FrEIA.framework as Ff
import FrEIA.modules as Fm
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import constants as const

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = 1-torch.exp(-torch.mean(x ** 2, dim=1, keepdim=True) * 0.5)
        max_out = 1-torch.exp(-torch.max(x ** 2, dim=1, keepdim=True)[0] * 0.5)
        att = torch.cat([avg_out, max_out], dim=1)
        att = self.conv1(att)
        att = self.sigmoid(att)
        x = x + x * att
        return x


class flow2(nn.Module):
    def __init__(self, in_channels,size,conv3x3_only,hidden_ratio,flow_steps):
        super(flow2, self).__init__()
        self.nf_flows = nn.ModuleList()

        self.nf_flows.append(
            nf_fast_flow_2(
                [in_channels,size,size],
                conv3x3_only=conv3x3_only,
                hidden_ratio=hidden_ratio,
                flow_steps=flow_steps,
                attention=SpatialAttention(),
            )
        )
    def forward(self, x):
        output, _ = self.nf_flows[0](x)
        return output

def subnet_conv_func(kernel_size, hidden_ratio,conv3x3_only,flow_steps,size):
    def subnet_conv(in_channels, out_channels):
        return nn.Sequential(
            flow2(in_channels,size,conv3x3_only,hidden_ratio,const.flow_li),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding="same"),
        )

    return subnet_conv
def subnet_conv_func_2(kernel_size, hidden_ratio,attention):
    def subnet_conv(in_channels, out_channels):
        hidden_channels = int(in_channels * hidden_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding="same"),
            attention,
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size, padding="same"),
        )

    return subnet_conv

def nf_fast_flow(input_chw, conv3x3_only, hidden_ratio, flow_steps,size, clamp=2.0):
    nodes = Ff.SequenceINN(*input_chw)
    for i in range(flow_steps):
        if i % 2 == 1 and not conv3x3_only:
            kernel_size = 1
        else:
            kernel_size = 3
        nodes.append(
            Fm.AllInOneBlock,
            subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio,conv3x3_only,flow_steps,size),
            affine_clamping=clamp,
            permute_soft=False,
        )
    return nodes
def nf_fast_flow_2(input_chw, conv3x3_only, hidden_ratio, flow_steps,attention, clamp=2.0):
    nodes = Ff.SequenceINN(*input_chw)
    for i in range(flow_steps):
        if i % 2 == 1 and not conv3x3_only:
            kernel_size = 1
        else:
            kernel_size = 3
        nodes.append(
            Fm.AllInOneBlock,
            subnet_constructor=subnet_conv_func_2(kernel_size, hidden_ratio,attention),
            affine_clamping=clamp,
            permute_soft=False,
        )
    return nodes

class SS_Dualflow(nn.Module):
    def __init__(
        self,
        backbone_name,
        flow_steps,
        input_size,
        conv3x3_only=False,
        hidden_ratio=1.0,
    ):
        super(SS_Dualflow, self).__init__()
        assert (
            backbone_name in const.SUPPORTED_BACKBONES
        ), "backbone_name must be one of {}".format(const.SUPPORTED_BACKBONES)

        if backbone_name in [const.BACKBONE_CAIT, const.BACKBONE_DEIT]:
            self.feature_extractor = timm.create_model(backbone_name, pretrained=True)
            channels = [768]
            scales = [16]
        else:
            self.feature_extractor = timm.create_model(
                backbone_name,
                pretrained=True,
                features_only=True,
                out_indices=[1, 2, 3],
            )
            channels = self.feature_extractor.feature_info.channels()
            scales = self.feature_extractor.feature_info.reduction()

            # for transformers, use their pretrained norm w/o grad
            # for resnets, self.norms are trainable LayerNorm
            self.norms = nn.ModuleList()
            for in_channels, scale in zip(channels, scales):
                self.norms.append(
                    nn.LayerNorm(
                        [in_channels, int(input_size / scale), int(input_size / scale)],
                        elementwise_affine=True,
                    )
                )

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.nf_flows = nn.ModuleList()
        for in_channels, scale in zip(channels, scales):
            self.nf_flows.append(
                nf_fast_flow(
                    [in_channels, int(input_size / scale), int(input_size / scale)],
                    conv3x3_only=conv3x3_only,
                    hidden_ratio=hidden_ratio,
                    flow_steps=const.flow_wai,
                    size=int(input_size / scale),
                )
            )
        self.input_size = input_size
        self.loss_focal =nn.MSELoss()


    def forward(self,augmented_image,anomaly_mask):
        self.feature_extractor.eval()
        features = self.feature_extractor(augmented_image)
        features = [self.norms[i](feature) for i, feature in enumerate(features)]

        flow_loss = 0
        outputs = []
        for i in range(len(features)):
            feature = features[i]
            output, log_jac_dets = self.nf_flows[i](feature)
            outputs.append(output)
        ret = {"flow_loss": flow_loss}

        anomaly_map_list = []
        b, c, w, h = anomaly_mask.size()
        for output in outputs:
            log_prob = -torch.mean(output**2, dim=1, keepdim=True) * 0.5
            prob = torch.exp(log_prob)
            a_map = F.interpolate(
                -prob,
                size=[self.input_size, self.input_size],
                mode="bilinear",
                align_corners=False,
            )
            anomaly_map_list.append(a_map)
        anomaly_map_list = torch.stack(anomaly_map_list, dim=-1)
        anomaly_map = torch.mean(anomaly_map_list, dim=-1)

        anomaly_loss =self.loss_focal(anomaly_map, anomaly_mask-1) *b*c*w*h

        ret["anomaly_loss"] = anomaly_loss

        ret["loss"] =  anomaly_loss
        ret["anomaly_map"] = anomaly_map

        return ret
