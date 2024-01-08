import os

import cv2
import numpy as np
import torch
import vision_transformer as vits
from simCLR.pic_pre_process import get_pic_random
from simsiam.sim_model.model_stage_1_vit import vit_simsiam
from src_ele.pic_opeeration import get_pic_distribute




######################################################################
# 这个方法的主要作用是
# 根据给出的 动态、静态 图像
# 计算出 图像的 储层类型预测 模型输入数据
########################################################################
def get_layerclassify_modelindata_from_dyna_stat_pic(pic_dyna, pic_stat, depth, RB_index=np.random.randint(0, 2)):
    pic_dyna = pic_dyna.reshape((1, pic_dyna.shape[-2], pic_dyna.shape[-1]))
    pic_stat = pic_stat.reshape((1, pic_stat.shape[-2], pic_stat.shape[-1]))
    # print(pic_dyna.shape)
    pic_all = np.append(pic_dyna, pic_stat, axis=0)

    pic_all_N1, RB_index = get_pic_random(pic_all, depth, RB_index)

    color_feature = get_pic_distribute(pic_all_N1[1], dist_length=9, min_V=0, max_V=1) * 2
    color_feature = np.append(color_feature, np.mean(pic_all_N1[1]))

    return pic_all_N1.astype(np.float32), color_feature.astype(np.float32)






# 图像分割任务 图像预处理部分 预设的一些东西
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    pretrained_weights = r''
else:
    pretrained_weights = r'D:/Data/target_answer/256-dyna-stat/checkpoint_VIT_8patch_56_0014.pth.tar'
image_size = (224, 224)
patch_size = 8
model = vits.__dict__['vit_small'](patch_size=patch_size, in_chans=2, num_classes=0)
# 2通道输入，8patch分割的  vision transformer模型构建
model_temp = vit_simsiam(in_chans=2, patch_size=8)
# 模型所有参数冻结
for p in model.parameters():
    p.requires_grad = False

model.eval()
model.to(device)

# 根据预训练的模型参数 预设vision transformer 模型参数
if os.path.isfile(pretrained_weights):
    state_dict = torch.load(pretrained_weights, map_location=device)
    model_temp.load_state_dict(state_dict['model_dict'])

    dict_temp = {}
    for name, param in model.named_parameters():
        for name_t, param_t in model_temp.named_parameters():
            if name_t.__contains__(name):
                dict_temp.update({name: param_t})
                break
    model.load_state_dict(dict_temp)
else:
    print('error pretrained_weights path')
    exit(0)
del model_temp


def get_attention_maps(imgs = np.array([])):
    imgs = imgs.reshape(1, 2, imgs.shape[-2], imgs.shape[-1])


    img = torch.from_numpy(imgs).float()
    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size
    attentions = model.get_last_selfattention(img.to(device))
    nh = attentions.shape[1]  # number of head       # nh = 6
    # # # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    attentions = attentions.reshape(nh, w_featmap, h_featmap)

    # print(attentions.shape)
    if torch.cuda.is_available():
        attentions = attentions.cpu().numpy()
    else:
        attentions = attentions.numpy()

    return attentions



######################################################################
# 这个方法的主要作用是
# 根据给出的 动态、静态 图像
# 计算出 图像的 图像分割任务 模型输入数据
########################################################################
def get_picsegmentation_modelindata_from_dyna_stat_pic(pic_dyna, pic_stat, depth, RB_index=np.random.randint(0, 2), pre_enhance=True):
    pic_dyna = pic_dyna.reshape((1, pic_dyna.shape[-2], pic_dyna.shape[-1]))
    pic_stat = pic_stat.reshape((1, pic_stat.shape[-2], pic_stat.shape[-1]))

    pic_all = np.append(pic_dyna, pic_stat, axis=0)

    pic_shape = (224, 224)
    if pre_enhance:
        pic_all_New, RB_index = get_pic_random(pic_all, depth, RB_index, pic_shape=pic_shape)
    else:
        pic_all_New = []
        pic_all_New.append(cv2.resize(pic_all[0, :, :], pic_shape)/256)
        pic_all_New.append(cv2.resize(pic_all[1, :, :], pic_shape)/256)
        pic_all_New = np.array(pic_all_New)

    pic_zoom = []
    mi_t = 1
    for i in range(4):
        pic_n_t = []
        for j in range(2):
            pic_n_t.append(cv2.resize(pic_all_New[j, :, :], dsize=(pic_shape[0] // mi_t, pic_shape[1] // mi_t),
                                      interpolation=cv2.INTER_AREA))
        pic_n_t = np.array(pic_n_t)
        mi_t = mi_t * 2
        # print(pic_n_t.shape, mi_t)
        pic_zoom.append(pic_n_t)

    attn = get_attention_maps(pic_all_New[:2, :, :])

    return pic_all_New, attn, pic_zoom[0], pic_zoom[1], pic_zoom[2], pic_zoom[3]