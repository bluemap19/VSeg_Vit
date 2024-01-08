import torch
import torch.nn as nn


# class unetUp(nn.Module):
#     def __init__(self, in_size, out_size):
#         super(unetUp, self).__init__()
#         # self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
#         # self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
#         self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=1, padding=0)
#         self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=1, padding=0)
#         # 上采样，放大两倍
#         self.up = nn.UpsamplingBilinear2d(scale_factor=2)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, inputs1, inputs2):
#         # print('inputs1 shape:{}, inputs2 shape:{}, '.format(inputs1.shape, inputs2.shape))
#         outputs = torch.cat([inputs1, self.up(inputs2)], 1)
#         outputs = self.conv1(outputs)
#         outputs = self.relu(outputs)
#         outputs = self.conv2(outputs)
#         outputs = self.relu(outputs)
#         return outputs



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
# 小型号的resnet block块
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# 大型号的resnet block块
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # 利用1x1卷积下降通道数
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        # 利用3x3卷积进行特征提取
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        # 利用1x1卷积上升通道数
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class model_S(nn.Module):
    def __init__(self, num_in_dim=384, num_out_dim=1):
        super(model_S, self).__init__()

        inter_dim = [256, 64]

        self.WV_m = nn.Sequential(nn.Conv2d(num_in_dim, inter_dim[0], kernel_size=3, padding=1),
                                        nn.ReLU(inplace=True),      # first layer
                                        # nn.BatchNorm2d(inter_dim[0]),

                                        Bottleneck(inter_dim[0], inter_dim[0]//4),
                                        Bottleneck(inter_dim[0], inter_dim[0]//4),

                                        nn.Conv2d(inter_dim[0], inter_dim[1], kernel_size=3, padding=1),
                                        nn.ReLU(inplace=True),      # second layer

                                        BasicBlock(inter_dim[1], inter_dim[1]),
                                        BasicBlock(inter_dim[1], inter_dim[1]),

                                        nn.Conv2d(inter_dim[1], num_out_dim, kernel_size=3, padding=1),            # 等于 (6): Linear(in_features=2048, out_features=2048, bias=True)
                                        nn.BatchNorm2d(num_out_dim)
                                  )      # output layer


    def forward(self, x):
        x = self.WV_m(x)

        return x


class UP_Cat_block(nn.Module):
    def __init__(self, in_size, out_size):
        super(UP_Cat_block, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
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

# 中号V字模型
class model_V(nn.Module):
    def __init__(self, num_in_dim=384, num_out_dim=1):
        super(model_V, self).__init__()
        self.pre_blk = nn.Conv2d(num_in_dim, 256, kernel_size=1, stride=1, bias=False)
        # self.base_block1 = BasicBlock(256, 256)
        self.base_block1 = Bottleneck(256, 64)
        self.down_sample = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(256)
        )
        self.base_block_down = nn.Sequential(Bottleneck(256, 64),
                                             Bottleneck(256, 64),)
        # self.base_block_down = nn.Sequential(Bottleneck(256, 64),
        #                                      # BasicBlock(256, 256),
        #                                      nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
        #                                      nn.BatchNorm2d(256),
        #                                      # BasicBlock(256, 256),
        #                                      Bottleneck(128, 32),)
        self.cat_blk = UP_Cat_block(256+256, 256)
        # self.base_block2 = BasicBlock(256, 256)
        self.base_block2 = Bottleneck(256, 64)
        self.blk2_to_end = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            BasicBlock(64, 64),
            nn.Conv2d(64, num_out_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_out_dim),
        )


    def forward(self, x):
        x = self.pre_blk(x)
        x = self.base_block1(x)
        x_down = self.down_sample(x)
        x_down = self.base_block_down(x_down)
        x = self.cat_blk(x, x_down)
        x = self.base_block2(x)
        x = self.blk2_to_end(x)
        return x

# 大号V字模型
class model_VL(nn.Module):
    def __init__(self, num_in_dim=384, num_out_dim=1):
        super(model_VL, self).__init__()
        self.pre_blk = nn.Conv2d(num_in_dim, 256, kernel_size=1, stride=1, bias=False)
        self.base_block1 = BasicBlock(256, 256)
        # self.base_block1 = Bottleneck(256, 64)
        self.down_sample = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(256)
        )
        self.base_block_down = nn.Sequential(BasicBlock(256, 256),
                                             nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False),
                                             nn.BatchNorm2d(128),
                                             BasicBlock(128, 128),)
        self.cat_blk = UP_Cat_block(256+128, 256)
        self.base_block2 = BasicBlock(256, 256)
        self.blk2_to_bl3 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.base_block3 = BasicBlock(64, 64)
        self.bl3_to_end = nn.Sequential(
            nn.Conv2d(64, num_out_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_out_dim),
        )

    def forward(self, x):
        x = self.pre_blk(x)
        x = self.base_block1(x)
        x_down = self.down_sample(x)
        x_down = self.base_block_down(x_down)
        x = self.cat_blk(x, x_down)
        x = self.base_block2(x)
        x = self.blk2_to_bl3(x)
        x = self.base_block3(x)
        x = self.bl3_to_end(x)
        return x


# # a = dataloader_ele_seg()
# # pic_all_New, attn, in_p_1, in_p_2, in_p_4, in_p_8 = a[0]
# # attn1 = torch.from_numpy(attn.reshape(1, 6, 28, 28))
# # in_p_1 = torch.from_numpy(in_p_1.reshape(1, 2, in_p_1.shape[-2], in_p_1.shape[-1])).float()
# # in_p_2 = torch.from_numpy(in_p_2.reshape(1, 2, in_p_2.shape[-2], in_p_2.shape[-1])).float()
# # in_p_4 = torch.from_numpy(in_p_4.reshape(1, 2, in_p_4.shape[-2], in_p_4.shape[-1])).float()
# # in_p_8 = torch.from_numpy(in_p_8.reshape(1, 2, in_p_8.shape[-2], in_p_8.shape[-1])).float()
# #
# # print(attn1.shape, in_p_1.shape, in_p_2.shape, in_p_4.shape, in_p_8.shape)
# x = torch.randn((2, 384, 9, 9))
# model = WV_model(num_in_dim=384, num_out_dim=1)
# answer = model(x)
# print(answer.shape)


# x = torch.randn((2, 256, 9, 9))
# e = Bottleneck(256, 64)
# # e = BasicBlock(256, 64)
# print(e)
# f = e(x)
# print(f.shape)