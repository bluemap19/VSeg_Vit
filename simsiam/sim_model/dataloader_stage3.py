import copy
import os

import cv2
import numpy as np

from simCLR.pic_pre_process import get_pic_random
from src_ele.dir_operation import traverseFolder_folder, traverseFolder
from torch.utils.data import Dataset

from src_ele.file_operation import get_ele_data_from_path

# 返回的是，同一个类别，即同一个文件夹下的两个不同的图像，用于第二阶段的，有监督的，对比学习
from src_ele.pic_opeeration import show_Pic, get_pic_distribute


class DatasetFolder(Dataset):
    def __init__(self, path=r'D:\1111\target_stage3'):
        super().__init__()
        self.list_all_folder = traverseFolder_folder(path)
        # print('all folder is :{}'.format(self.list_all_folder))
        self.list_file = []
        for i in self.list_all_folder:
            self.list_file.append(traverseFolder(i))
            # print('folder:{} contains file num:{}'.format(i, len(self.list_file[-1])))
        self.list_file_all = traverseFolder(path)
        self.length = len(self.list_file_all)

        self.RB_index = -1
        # print('datapath:{}'.format(path))
        # print('all file num is :{}'.format(len(self.list_file_all)))

    def __getitem__(self, index):
        class_num = int(self.list_file_all[index].split('/')[-2])

        # 把地层识别任务 只设置成 判断是不是裂缝型 其他地层类型一律设置为 0：干层
        if class_num == 0:
            class_num = np.array([0, 0, 0, 0])
        elif class_num == 1:
            class_num = np.array([1, 0, 0, 1])
        elif class_num == 2:
            class_num = np.array([0, 1, 0, 2])
        elif class_num == 3:
            class_num = np.array([1, 1, 0, 3])
        elif class_num == 4:
            class_num = np.array([0, 0, 1, 4])
        elif class_num == 5:
            class_num = np.array([1, 0, 1, 5])
        elif class_num == 6:
            class_num = np.array([0, 1, 1, 6])
        elif class_num == 7:
            class_num = np.array([1, 1, 1, 7])
        else:
            print('class num error:{}'.format(class_num))
            exit(0)

        path_temp = self.list_file_all[index]
        path_temp_stat = ''
        # print(path_temp)

        if path_temp.__contains__('dyna'):
            path_temp_stat = path_temp.replace('dyna', 'stat')
        elif path_temp.__contains__('stat'):
            path_temp_stat = path_temp
            path_temp = path_temp_stat.replace('stat', 'dyna')
        else:
            print('what is the path:{}'.format(path_temp))
            exit(0)


        if os.path.isfile(path_temp_stat):
            pass
        else:
            print('cont find vice pic:{}'.format(path_temp_stat))
            exit(0)

        pic_dyna, depth = get_ele_data_from_path(path_temp)
        pic_stat, depth = get_ele_data_from_path(path_temp_stat)

        # pic_all = np.zeros((2, pic_dyna.shape[0], pic_dyna.shape[1]))
        pic_dyna = pic_dyna.reshape((1, pic_dyna.shape[0], pic_dyna.shape[1]))
        pic_stat = pic_stat.reshape((1, pic_stat.shape[0], pic_stat.shape[1]))
        # print(pic_dyna.shape)
        pic_all = np.append(pic_dyna, pic_stat, axis=0)

        pic_all_N1, self.RB_index = get_pic_random(pic_all, depth, self.RB_index)

        # pic_all_N1_temp = copy.deepcopy(pic_all_N1)
        # pic_color_dis = cv2.resize(pic_all_N1_temp[1], (20, 1))
        # color_feature = np.append(color_feature, np.mean(pic_all_N1_temp[1]))
        # # print(color_feature, np.mean(color_feature))

        color_feature = get_pic_distribute(pic_all_N1[1], dist_length=9, min_V=0, max_V=1)
        color_feature = np.append(color_feature, np.mean(pic_all_N1[1]))
        # color_feature = np.append(color_feature, pic_color_dis)
        # print(color_feature, np.sum(color_feature[:9]))

        return pic_all_N1.astype(np.float32), class_num, color_feature.astype(np.float32)

    def __len__(self):
        return self.length

        # 或者return len(self.trg),  src 和 trg 长度一样

# a = DatasetFolder(r'D:\Data\target_stage3_small_p\train')
# print(a.length)
# print(a[10][-2].shape)
# show_Pic([a[10][0][0]*256, a[10][0][1]*256], pic_order='12', pic_str=[], save_pic=False, path_save='')
# color_feature = np.zeros((10))
