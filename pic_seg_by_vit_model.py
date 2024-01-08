import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import vision_transformer as vits
from simsiam.sim_model.model_stage_1_vit import vit_simsiam


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x



class Segmenter(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = vits.__dict__['vit_small'](patch_size=8, in_chans=2, num_classes=0)
        model_temp = vit_simsiam()
        resume = r'D:\Data\target_answer\250X250\checkpoint_res50_batch240_dim15_epoch0016.pth.tar'
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False

        # 判断模型文件是否存在
        if os.path.isfile(resume):
            # 使用GPU运行
            if torch.cuda.is_available():
                DEVICE = torch.device("cuda:" + str(torch.cuda.current_device()))
                print('putting model on single GPU:{}'.format(DEVICE))

                # state_dict = torch.load(resume, map_location=DEVICE)
                # model_temp.load_state_dict(state_dict['model_dict'])
                #
                # # 单GPU该怎么载入模型
                # self.encoder.cuda()
                #
                # print('load encoder from:{}'.format(resume))
                # # self.encoder = self.encoder.to(DEVICE)
                # checkpoint = torch.load(resume)
                # self.encoder.load_state_dict(checkpoint['model_dict'])
            else:
                DEVICE = torch.device("cpu")

            state_dict = torch.load(resume, map_location=DEVICE)
            model_temp.load_state_dict(state_dict['model_dict'])
            dict_temp = {}
            for name, param in self.encoder.named_parameters():
                for name_t, param_t in model_temp.named_parameters():
                    if name_t.__contains__(name):
                        dict_temp.update({name: param_t})
                        break

            self.encoder.load_state_dict(dict_temp)

                # print('CPU processing:{}'.format(DEVICE))
                # checkpoint = torch.load(resume, map_location=DEVICE)
                # self.encoder.to(DEVICE)
                # self.encoder.load_state_dict(checkpoint['model_dict'])  # model_dict

                # args.start_epoch = checkpoint['epoch']
                # optimizer.load_state_dict(checkpoint['optimizer'])
                # print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))
            exit(0)


    def forward(self, im):
        H_ori, W_ori = im.size(-2), im.size(-1)
        # 即padding是在图像的边缘增加几个像素，目的是保持特征图不要太小，添加的个数与卷积核有关
        # im = padding(im, self.patch_size)
        H, W = im.size(-2), im.size(-1)

        # x = self.encoder.get_last_selfattention(im)

        x, attn_weights = self.encoderget_selfattention_and_feature(im)

        x = self.decoder(x, attn_weights)
        logits = self.segmentation_head(x)
        return logits


        # # remove CLS/DIST tokens for decoding
        # num_extra_tokens = 1
        # print('x shape is :{}'.format(x.shape))
        # x = x[:, num_extra_tokens:]
        # print('x shape is :{}'.format(x.shape))
        #
        # masks = self.decoder(x, (H, W))
        #
        # masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        # masks = unpadding(masks, (H_ori, W_ori))
        #
        # return masks

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)


# Segmenter()