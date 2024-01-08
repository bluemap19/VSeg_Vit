import torch
import torch.nn as nn

from simsiam.ele_seg.dataloader_ele_seg import dataloader_ele_seg


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        # self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=1, padding=0)
        # 上采样，放大两倍
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        # print('inputs1 shape:{}, inputs2 shape:{}, '.format(inputs1.shape, inputs2.shape))
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


class Unet_ele_seg(nn.Module):
    def __init__(self, num_classes=1):
        super(Unet_ele_seg, self).__init__()
        # if backbone == 'vgg':
        #     self.vgg = VGG16(pretrained=pretrained)
        #     in_filters = [192, 384, 768, 1024]
        # elif backbone == "resnet50":
        #     self.resnet = resnet50(pretrained=pretrained)
        #     in_filters = [192, 512, 1024, 3072]
        # else:
        #     raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        in_filters = [8, 14, 23]
        out_filters = [12, 21, 30]

        # # upsampling
        # # 64,64,512
        # self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])

        # if backbone == 'resnet50':
        #     self.up_conv = nn.Sequential(
        #         nn.UpsamplingBilinear2d(scale_factor=2),
        #         nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
        #         nn.ReLU(),
        #         nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
        #         nn.ReLU(),
        #     )
        # else:
        #     self.up_conv = None

        self.final = nn.Conv2d(out_filters[2], num_classes, 1)

        # self.backbone = backbone

    def forward(self, atts, in_4, in_2, in_1):
        # if self.backbone == "vgg":
        #     [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        # elif self.backbone == "resnet50":
        #     [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        # up4 = self.up_concat4(feat4, feat5)
        up1 = self.up_concat1(in_4, atts)
        up2 = self.up_concat2(in_2, up1)
        up3 = self.up_concat3(in_1, up2)

        # if self.up_conv != None:
        #     up1 = self.up_conv(up1)

        final = self.final(up3)

        return final

    # def freeze_backbone(self):
    #     if self.backbone == "vgg":
    #         for param in self.vgg.parameters():
    #             param.requires_grad = False
    #     elif self.backbone == "resnet50":
    #         for param in self.resnet.parameters():
    #             param.requires_grad = False
    #
    # def unfreeze_backbone(self):
    #     if self.backbone == "vgg":
    #         for param in self.vgg.parameters():
    #             param.requires_grad = True
    #     elif self.backbone == "resnet50":
    #         for param in self.resnet.parameters():
    #             param.requires_grad = True


# a = dataloader_ele_seg()
# pic_all_New, attn, in_p_1, in_p_2, in_p_4, in_p_8 = a[0]
# attn1 = torch.from_numpy(attn.reshape(1, 6, 28, 28))
# in_p_1 = torch.from_numpy(in_p_1.reshape(1, 2, in_p_1.shape[-2], in_p_1.shape[-1])).float()
# in_p_2 = torch.from_numpy(in_p_2.reshape(1, 2, in_p_2.shape[-2], in_p_2.shape[-1])).float()
# in_p_4 = torch.from_numpy(in_p_4.reshape(1, 2, in_p_4.shape[-2], in_p_4.shape[-1])).float()
# in_p_8 = torch.from_numpy(in_p_8.reshape(1, 2, in_p_8.shape[-2], in_p_8.shape[-1])).float()
#
# print(attn1.shape, in_p_1.shape, in_p_2.shape, in_p_4.shape, in_p_8.shape)
#
# model = Unet_ele_seg(num_classes=1)
# answer = model(attn1, in_p_4, in_p_2, in_p_1)
# print(answer.shape)
