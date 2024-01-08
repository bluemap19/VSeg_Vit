import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from simCLR.pic_pre_process import get_pic_random
from simsiam.sim_model.model_stage_1_vit import vit_simsiam
from src_ele.dir_operation import traverseFolder
from src_ele.file_operation import get_ele_data_from_path
import vision_transformer as vits
# import torch.nn as nn



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    pretrained_weights = r''
else:
    pretrained_weights = r'D:/Data/target_answer/256-dyna-stat/checkpoint_VIT_8patch_56_0014.pth.tar'
image_size = (224, 224)
patch_size = 8
model = vits.__dict__['vit_small'](patch_size=patch_size, in_chans=2, num_classes=0)
model_temp = vit_simsiam()
for p in model.parameters():
    p.requires_grad = False

model.eval()
model.to(device)

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
    # # torch.unsqueeze扩展维度
    # # 返回一个新的张量，对输入的既定位置插入维度 1
    # attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()


# 返回的是，文件夹下的，随机一张图片的，经过处理后的两张图像增强照片，用于阶段1的无监督学习的对比学习
class dataloader_ele_seg(Dataset):
    def __init__(self, path=r'D:\Data\img_seg_data_in\train'):
        super().__init__()
        self.list_all_file = traverseFolder(path)
        self.length = len(self.list_all_file)//3
        self.RB_index = np.random.randint(0, 2)

    def __getitem__(self, index):
        path_temp = self.list_all_file[index*3]
        path_temp_stat = ''
        # print(path_temp)

        if path_temp.__contains__('dyna'):
            path_temp_stat = path_temp.replace('dyna', 'stat')
            path_temp_mask = path_temp.replace('dyna', 'mask')
        elif path_temp.__contains__('stat'):
            path_temp_stat = path_temp
            path_temp = path_temp_stat.replace('stat', 'dyna')
            path_temp_mask = path_temp_stat.replace('stat', 'mask')

        # print(path_temp)

        pic_dyna, depth = get_ele_data_from_path(path_temp)
        pic_stat, depth = get_ele_data_from_path(path_temp_stat)
        pic_mask, _ = get_ele_data_from_path(path_temp_mask)
        pic_mask = cv2.resize(pic_mask, (pic_dyna.shape[1], pic_dyna.shape[0]))
        # print(pic_dyna.shape, pic_stat.shape, pic_mask.shape)

        # print(pic_mask.shape)

        # show_Pic([pic_dyna, pic_stat], pic_order='12', pic_str=['', ''], save_pic=False, path_save='')

        pic_dyna = pic_dyna.reshape((1, pic_dyna.shape[0], pic_dyna.shape[1]))
        pic_stat = pic_stat.reshape((1, pic_stat.shape[0], pic_stat.shape[1]))
        pic_mask = pic_mask.reshape((1, pic_mask.shape[0], pic_mask.shape[1]))

        self.pic_all = np.append(pic_dyna, pic_stat, axis=0)
        self.pic_all = np.append(self.pic_all, pic_mask, axis=0)

        # print(self.pic_all.shape)

        pic_shape = (224, 224)
        pic_all_New, self.RB_index = get_pic_random(self.pic_all, depth, self.RB_index, pic_shape=pic_shape)

        pic_zoom = []
        mi_t = 1
        for i in range(4):
            pic_n_t = []
            for j in range(2):
                pic_n_t.append(cv2.resize(pic_all_New[j, :, :], dsize=(pic_shape[0]//mi_t, pic_shape[1]//mi_t), interpolation=cv2.INTER_AREA))
            pic_n_t = np.array(pic_n_t)
            mi_t = mi_t * 2
            # print(pic_n_t.shape, mi_t)
            pic_zoom.append(pic_n_t)

        attn = get_attention_maps(pic_all_New[:2, :, :])

        # all_feature_list = model_temp.

        return pic_all_New, attn, pic_zoom[0], pic_zoom[1], pic_zoom[2], pic_zoom[3]

    def __len__(self):
        return self.length
        # 或者return len(self.trg), src和trg长度一样

# a = dataloader_ele_seg()
# print('pic_seg_data_train_in num is :{}'.format(a.length))
# pic_all_New, attn, pic_zoom = a[0]
# print(pic_all_New.shape, attn.shape, pic_zoom[0].shape, pic_zoom[1].shape, pic_zoom[2].shape, pic_zoom[3].shape)
# # # index = np.random.randint(0, 100)
# a = dataloader_ele_DINO(path=r'D:\Data\target_stage1_small')
# index_random = np.random.randint(0, a.length)
# # index_random = 2099
# print(index_random)
# a[index_random]
# # for i in range(a.length):
# #     a[i+np.random.randint(0, 1000)]
#     # show_Pic([a[i][0]*256, a[i][1]*256], pic_order='12', pic_str=[], save_pic=False, path_save='')
# # print(a.length)
# # print(a[0][0].shape)

# (a.length)
# print(a[0][0].shape)

