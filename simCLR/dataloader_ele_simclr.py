import copy
import numpy as np
from torch.utils.data import Dataset

from simCLR.pic_pre_process import get_pic_random
from src_ele.dir_operation import traverseFolder
from src_ele.file_operation import get_ele_data_from_path





class dataloader_ele_simclr(Dataset):
    def __init__(self, path=r'img_temp', data_len=-1):
        super().__init__()
        self.list_all_file = traverseFolder(path)
        self.length = len(self.list_all_file)
    def __getitem__(self, index):

        img, depth = get_ele_data_from_path(self.list_all_file[index])
        label = 0.0
        img_r = get_pic_random(img, depth).astype(np.float64)
        img_l = get_pic_random(img, depth).astype(np.float64)

        return img_r, img_l, label

    def __len__(self):
        return self.length

        # 或者return len(self.trg), src和trg长度一样

def dataset_collate_ele_repair(batch):
    list_1 = []
    list_2 = []
    list_3 = []
    for pic1, pic2, pic3  in batch:
        list_1.append(pic1)
        list_2.append(pic2)
        list_3.append(pic3)
    list_1 = np.array(list_1)
    list_2 = np.array(list_2)
    list_3 = np.array(list_3)
    return list_1, list_2, list_3
    # return list_1, list_2

# a = dataloader_ele_simclr()
# print(a.length)
# print(a[0])
# print(a[1000])
# print(a[1001])
# print(a[1002])
# print(a[1003][0])
# print(a[2000][1])
# print(a[2000][2])
#
# print(np.mean(a[20000][2]))
# print(np.mean(a[20000][3]))
# for i in range(a.length):
#     print(np.mean(a[i][0]))
# # for i in range(a.length):
#     print(a[i][1])