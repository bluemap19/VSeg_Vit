import copy
import os

import cv2
import torch
from simsiam.ele_seg_2.dataloader_ele_seg import dataloader_bottle, get_windows_pic_from_feature_maps_FULLY, \
    dataloader_up1, dataloader_base, dataloader_up2, dataloader_up3
from simsiam.ele_seg_2.model_fracture_seg import model_VL, BasicBlock, model_V, model_S
from simsiam.ele_seg_2.resnet import resnet50
from src_ele.dir_operation import traverseFolder
from src_ele.file_operation import get_ele_data_from_path
from src_ele.pic_opeeration import show_Pic
import numpy as np

# path_t = r'D:\Data\pic_seg_choices\data_paper'
# a = dataloader_base(path=path_t, pre_process=False)
# print('pic_seg_data_train_in num is :{}'.format(a.length))
#
#
# pic_index = [3, 6, 7, 8, 9]
# channel_index = [11, 67, 136, 310, 331, 343, 382]
# for i in pic_index:
#     pic_all_New, all_feature_list = a[i]
#     b_p = [pic_all_New[0, :, :] * 256, pic_all_New[1, :, :] * 256, pic_all_New[-1, :, :] * 256]
#
#     name_1 = '{}_dyna.png'.format(i)
#     cv2.imwrite(name_1, pic_all_New[0, :, :] * 256)
#     name_2 = '{}_stat.png'.format(i)
#     cv2.imwrite(name_2, pic_all_New[1, :, :] * 256)
#     name_3 = '{}_mask.png'.format(i)
#     cv2.imwrite(name_3, pic_all_New[2, :, :] * 256)
#
#     for j in channel_index:
#         # b = []
#         b = copy.deepcopy(b_p)
#         for k in range(all_feature_list.shape[0]):
#             name_t = '{}_channel_{}_layer_{}_feature.png'.format(i, j, k)
#             cv2.imwrite(name_t, all_feature_list[k][0][j] * 256)
#             b.append(all_feature_list[k][0][j] * 256)
# for i in range(a.length):
#     pic_all_New, all_feature_list = a[i]
#     print('pic_all_New:{}, all_feature_list:{}'.format(pic_all_New.shape, all_feature_list.shape))
#
#     # 展示不同layer层之间，feature map 的特征递进过程
#     # 好像index为8、52的通道，其特征递进过程比较明显
#     str = ['img_dyna', 'img_stat', 'img_mask',
#            'layer:0', 'layer:2', 'layer:4', 'layer:6',
#            'layer:8', 'layer:10', 'layer:12']
#
#     # channel_list = [8, 331, 335, 343]
#     # channel_list = [42, 64, 310]
#     # channel_list = [43, 67, 196, 382]
#     # channel_list = [83, 160, 256]
#     channel_list = [67, 136, 310, 331, 343, 382]
#
#     b_p = [pic_all_New[0, :, :]*256, pic_all_New[1, :, :]*256, pic_all_New[-1, :, :]*256]
#
#     for j in range(len(channel_list)):
#         # all_feature_list.shape[2]
#
#         b = copy.deepcopy(b_p)
#         for k in range(all_feature_list.shape[0]//2):
#             b.append(all_feature_list[k*2][0][channel_list[j]]*256)
#
#         print('now channel is :{}'.format(channel_list[j]))
#         show_Pic(b, pic_order='33', pic_str=str)
    # exit(0)


# pic_all_New, all_feature_list, embedding_pic_split, mask_split = a[9]
# # pic_all_New, embedding_pic_split, mask_split = a[9]
# print('pic_all_New:{}, all_feature_list:{}, embedding_pic_split:{}, mask_split:{}'.format(
#     pic_all_New.shape, all_feature_list.shape, embedding_pic_split.shape, mask_split.shape))
# show_Pic([pic_all_New[0, :, :]*256, pic_all_New[1, :, :]*256], '12', ['img.dyna', 'img.stat'])
# pic_all_New:(3, 224, 224), all_feature_list:(12, 1, 384, 28, 28), embedding_pic_split:(12, 16, 384, 14, 14), mask_split:(12, 16, 14, 14)


# # 查看12个通道，哪一个通道是顶层，哪一个通道时底层, 以及哪一个通道的效果更好
# # 0是顶层，-1是底层
# for i in range(all_feature_list.shape[0]):
#     a = []
#     for j in range(9):
#         a.append(all_feature_list[i][0][j+50]*256)
#     show_Pic(a, pic_order='33')


# # 展示不同layer层之间，feature map 的特征递进过程
# # 好像index为8、52的通道，其特征递进过程比较明显
# a = []
# b = []
# # c = np.random.randint(9, 383)
# c = 52
# print('random gate:{}'.format(c))
# for i in range(all_feature_list.shape[0]):
#     a.append(all_feature_list[i][0][8]*256)
#     b.append(all_feature_list[i][0][c]*256)
# str = ['layer:1', 'layer:2', 'layer:3', 'layer:4',
#        'layer:5', 'layer:6', 'layer:7', 'layer:8',
#        'layer:9', 'layer:10', 'layer:11', 'layer:12']
# show_Pic(a, pic_order='34', pic_str=str)
# show_Pic(b, pic_order='34', pic_str=str)


# # # 查看图像的分割效果
# # a = []
# # a.append(pic_all_New[0, :, :]*256)
# # a.append(pic_all_New[1, :, :]*256)
# # a.append(pic_all_New[2, :, :]*256)
# # for i in range(6):
# #     pei = embedding_pic_split.shape[1]//6
# #     a.append(embedding_pic_split[-1, i*pei, 0, :, :]*256)
# #     a.append(mask_split[0, i*pei, :, :]*256)
# # show_Pic(a, pic_order='35')
#
# # show_Pic([pic_all_New[0, :, :]*256, pic_all_New[1, :, :]*256], '12', ['img.dyna', 'img.stat'])
# in_pic_split, mask_out = get_windows_pic_from_feature_maps_FULLY(all_feature_list[8], pic_all_New[-1, :, :], windows_length=18)
# print(in_pic_split.shape, mask_out.shape)
# show_Pic([pic_all_New[0, :, :]*256, pic_all_New[-1, :, :]*256,
#           in_pic_split[0, 0, :, :]*256, in_pic_split[1, 0, :, :]*256,
#           in_pic_split[2, 0, :, :]*256, in_pic_split[3, 0, :, :]*256,
#           mask_out[0, :, :]*256, mask_out[1, :, :]*256,
#           mask_out[2, :, :]*256, mask_out[3, :, :]*256], '25')

# # 展示dataloader_up1函数输出效果
# b = dataloader_up1(windows_length=28, windows_step=7)
# for i in range(10):
#     pic_all_New, feature_in, feature_in_split, mask_org_split = b[i]
#     print(pic_all_New.shape, feature_in.shape, feature_in_split.shape, mask_org_split.shape)
#     show_Pic([pic_all_New[0, :, :]*256, pic_all_New[1, :, :]*256, pic_all_New[2, :, :]*256,
#               feature_in[0, 0, :, :]*256, feature_in_split[0, 0, :, :] * 256, feature_in_split[1, 0, :, :] * 256,
#               feature_in_split[2, 0, :, :]*256, feature_in_split[3, 0, :, :] * 256, feature_in_split[4, 0, :, :] * 256,
#               feature_in_split[-3, 0, :, :]*256, feature_in_split[-2, 0, :, :] * 256, feature_in_split[-1, 0, :, :] * 256], pic_order='43')


# model_WV = WV_model(384, 1)
# a = torch.randn((3, 384, 28, 28))
# print(model_WV(a).shape)

# model_W = W_chip_model(384, 1)
# # model_red50 = resnet50()
# print(model_W)
# # print(model_red50)
# a = torch.randn((3, 384, 28, 28))
# b = model_W(a)
# print(b.shape)


# c = BasicBlock(256, 256)
# d = c(a)
# print(d.shape)

# model = W_chip_model_2(384, 1)
# a = torch.randn((3, 384, 28, 28))
# b = model(a)
# print(b.shape)

# import vision_transformer as vits
# from simsiam.sim_model.model_stage_1_vit import vit_simsiam
#
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# # 配置 训练好的模型参数
# if torch.cuda.is_available():
#     pretrained_weights = r''
# else:
#     pretrained_weights = r'D:/Data/target_answer/256-dyna-stat/checkpoint_VIT_8patch_56_0014.pth.tar'
#     # pretrained_weights = r'D:\Data\target_answer\vit_small-64\checkpoint_vit_64_0014.pth.tar'
# image_size = (224, 224)
# patch_size = 8
# model = vits.__dict__['vit_small'](patch_size=patch_size, in_chans=2, num_classes=0)
# model_temp = vit_simsiam()
# for p in model.parameters():
#     p.requires_grad = False
#
# if os.path.isfile(pretrained_weights):
#     state_dict = torch.load(pretrained_weights, map_location=device)
#     model_temp.load_state_dict(state_dict['model_dict'])
#
#     dict_temp = {}
#     for name, param in model.named_parameters():
#         for name_t, param_t in model_temp.named_parameters():
#             if name_t.__contains__(name):
#                 dict_temp.update({name: param_t})
#                 break
#     model.load_state_dict(dict_temp)
# else:
#     print('error pretrained_weights path')
#     exit(0)
# # model.eval()就是帮我们一键搞定Dropout层和batch normalization层
# model.eval()
# model.to(device)
# del model_temp
# a = torch.randn((3, 2, 224, 224))
# print(model(a))



# a = dataloader_up2()
# pic_all_New, feature_in, feature_in_split, mask_org_split = a[0]
# print(pic_all_New.shape, feature_in.shape, feature_in_split.shape, mask_org_split.shape)




a = dataloader_up3()
for i in range(a.length):
    pic_all_New, feature_in, feature_in_split, mask_org_split = a[i]
    print(pic_all_New.shape, feature_in.shape, feature_in_split.shape, mask_org_split.shape)

exit(0)