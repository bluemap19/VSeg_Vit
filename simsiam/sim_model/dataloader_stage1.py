import copy
import numpy as np
from torch.utils.data import Dataset
from simCLR.pic_pre_process import get_pic_random
from src_ele.dir_operation import traverseFolder
from src_ele.file_operation import get_ele_data_from_path
from src_ele.pic_opeeration import show_Pic

# 返回的是，文件夹下的，随机一张图片的，经过处理后的两张图像增强照片，用于阶段1的无监督学习的对比学习
class dataloader_ele_my(Dataset):
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

        # pic_all = np.zeros((2, pic_dyna.shape[0], pic_dyna.shape[1]))
        pic_dyna = pic_dyna.reshape((1, pic_dyna.shape[0], pic_dyna.shape[1]))
        pic_stat = pic_stat.reshape((1, pic_stat.shape[0], pic_stat.shape[1]))
        # print(pic_dyna.shape)
        pic_all_1 = np.append(pic_dyna, pic_stat, axis=0)
        # print(pic_all.shape)
        pic_all_2 = copy.deepcopy(pic_all_1)
        # show_Pic([pic_all[0], pic_all[1]], pic_order='12', pic_str=[], save_pic=False, path_save='')
        pic_all_N1, RB_index = get_pic_random(pic_all_1, depth)
        pic_all_N2, RB_index = get_pic_random(pic_all_2, depth, RB_index)
        # pic_all_N1[1, :, :] = pic_all_N1[0, :, :]
        # pic_all_N2[1, :, :] = pic_all_N2[0, :, :]

        # pic_list = []
        # label = []
        # for i in range(10):
        #     pic_list.append(get_pic_random(pic_all, depth).astype(np.float32)/256)
        #     label.append(0)

        # print(pic_all_N1.shape)
        # show_Pic([pic_all[0]*256, pic_all[1]*256, pic_all_N1[0]*256, pic_all_N2[0]*256], pic_order='14', pic_str=[], save_pic=False, path_save='')

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
        return pic_all_N1, pic_all_N2, label
        # return img_r, img_l, label
        # return pic_list, label

    def __len__(self):
        return self.length

        # 或者return len(self.trg), src和trg长度一样


# # # index = np.random.randint(0, 100)
# a = dataloader_ele_my(path=r'D:\Data\target_stage1_small_big_mix')
# index_random = np.random.randint(0, a.length)
# # # index_random = 2099
# print(index_random)
# # a[index_random]
# for i in range(a.length):
#     temp = a[i]
#     show_Pic([temp[0][0]*256, temp[0][1]*256, temp[1][0]*256, temp[1][1]*256], pic_order='22', pic_str=[], save_pic=False, path_save='')
# # print(a.length)
# # print(a[0][0].shape)

# (a.length)
# print(a[0][0].shape)

