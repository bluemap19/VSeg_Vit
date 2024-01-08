import copy

import cv2
import numpy as np
from torch.utils.data import Dataset
from simCLR.pic_pre_process import get_pic_random, get_pic_random_VIT_teacher, \
    get_pic_random_VIT_student
from src_ele.dir_operation import traverseFolder
from src_ele.file_operation import get_ele_data_from_path
from src_ele.pic_opeeration import show_Pic, pic_scale_normal


# 返回的是，文件夹下的，随机一张图片的，经过处理后的两张图像增强照片，用于阶段1的无监督学习的对比学习
class dataloader_ele_DINO(Dataset):
    def __init__(self, path=r'D:\Data\target_uns_class'):
        super().__init__()
        self.list_all_file = traverseFolder(path)
        self.length = len(self.list_all_file)//2

    def __getitem__(self, index):
        path_temp = self.list_all_file[index*2]
        path_temp_stat = ''
        # print(path_temp)

        if path_temp.__contains__('dyna'):
            path_temp_stat = path_temp.replace('dyna', 'stat')
        elif path_temp.__contains__('stat'):
            path_temp_stat = path_temp
            path_temp = path_temp_stat.replace('stat', 'dyna')

        pic_dyna, depth = get_ele_data_from_path(path_temp)
        pic_stat, depth = get_ele_data_from_path(path_temp_stat)

        # show_Pic([pic_dyna, pic_stat], pic_order='12', pic_str=['', ''], save_pic=False, path_save='')

        pic_dyna = pic_dyna.reshape((1, pic_dyna.shape[0], pic_dyna.shape[1]))
        pic_stat = pic_stat.reshape((1, pic_stat.shape[0], pic_stat.shape[1]))
        pic_all = np.append(pic_dyna, pic_stat, axis=0)

        pic_all_N1, RB_index = get_pic_random_VIT_teacher(pic_all, depth)
        pic_all_N2, RB_index = get_pic_random_VIT_teacher(pic_all, depth, RB_index)
        # show_Pic([pic_all[0], pic_all[1]], pic_order='12', pic_str=[], save_pic=False, path_save='')
        # pic_all_N1[1, :, :] = pic_all_N1[0, :, :]
        # pic_all_N2[1, :, :] = pic_all_N2[0, :, :]

        # print(pic_all_N1.shape)


        pic_shape = (224, 224)
        pic_list = [pic_scale_normal(pic_all_N1, pic_shape)/256, pic_scale_normal(pic_all_N2, pic_shape)/256]
        for i in range(4):
            pic_t, RB_index = get_pic_random_VIT_student(pic_all_N1, depth, RB_index)
            pic_list.append(pic_t/256)
            pic_t, RB_index = get_pic_random_VIT_student(pic_all_N2, depth, RB_index)
            pic_list.append(pic_t/256)


        # # print(pic_all_N1.shape)
        # show_Pic([pic_list[0][0]*256, pic_list[0][1]*256, pic_list[1][0]*256,
        #           pic_list[2][0]*256, pic_list[3][0]*256, pic_list[4][0]*256,
        #           pic_list[5][0]*256, pic_list[6][0]*256, pic_list[7][0]*256], pic_order='33', save_pic=False)

        # img, depth = get_ele_data_from_path(path_temp)
        # # print(img.shape, depth.shape)
        label = 0.0
        # img_r = get_pic_random(img, depth).astype(np.float32)
        # img_l = get_pic_random(img, depth).astype(np.float32)
        # # # img_r_2 = np.zeros((3, img_r.shape[0], img_r.shape[1]))
        # # # img_l_2 = np.zeros((3, img_r.shape[0], img_r.shape[1]))
        # # img_r_2 = np.array([img_r, img_r, img_r]).astype(np.float32)
        # # img_l_2 = np.array([img_l, img_l, img_l]).astype(np.float32)

        # # return img_r_2, img_l_2, label
        return pic_list, label
        # return img_r, img_l, label
        # return pic_list, label

    def __len__(self):
        return self.length

        # 或者return len(self.trg), src和trg长度一样


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

