import numpy as np

from simCLR.pic_pre_process import get_pic_random
from src_ele.dir_operation import traverseFolder_folder, traverseFolder
from torch.utils.data import Dataset

from src_ele.file_operation import get_ele_data_from_path

# 返回的是，同一个类别，即同一个文件夹下的两个不同的图像，用于第二阶段的，有监督的，对比学习
class DatasetFolder(Dataset):
    def __init__(self, path=r'D:\Data\target_classed'):
        super().__init__()
        self.list_all_folder = traverseFolder_folder(path)
        # print('all folder is :{}'.format(self.list_all_folder))
        self.list_file = []
        for i in self.list_all_folder:
            self.list_file.append(traverseFolder(i))
            # print('folder:{} contains file num:{}'.format(i, len(self.list_file[-1])))
        self.list_file_all = traverseFolder(path)
        self.length = len(self.list_file_all)
        # print('all file num is :{}'.format(len(self.list_file_all)))

    def __getitem__(self, index):
        class_num = int(self.list_file_all[index].split('/')[-2])
        # print('current file class is :{}'.format(class_num))
        list_file_temp = traverseFolder(self.list_all_folder[class_num])
        index_temp = np.random.randint(0, len(list_file_temp))
        # print('all folder num is :{}, random file index is :{}'.format(len(list_file_temp), index_temp))

        img_R, depth_r = get_ele_data_from_path(self.list_file_all[index])
        img_L, depth_l = get_ele_data_from_path(list_file_temp[index_temp])

        img_r = get_pic_random(img_R, depth_r).astype(np.float32)
        img_l = get_pic_random(img_L, depth_l).astype(np.float32)

        return img_r, img_l, class_num

    def __len__(self):
        return self.length

        # 或者return len(self.trg),  src 和 trg 长度一样
# a = DatasetFolder()
# print(a.length)
# print(a[7][-2])