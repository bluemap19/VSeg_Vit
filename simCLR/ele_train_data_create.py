import numpy as np

from src_ele import dir_operation
from src_ele.pic_opeeration import show_Pic
from train_org_model.ELE_data import Ele_data

class get_spilt_pic:
    def __init__(self, folder_path = r'D:/Data/target_3', data_len = -1):
        super().__init__()

        # 获取文件夹内所有的 文件夹 信息，每一个文件夹都包含一口井的所有信息
        folder_list = dir_operation.traverseFolder_folder(folder_path)

        # 选定本次处理那些文件夹
        target_folder_list = ['LG7-8', 'LG7-4', 'LG7-16', 'LG701', 'LG701-H1', 'LN11-4', 'LN11-5']
        target_folder = []
        for i in range(len(folder_list)):
            for j in range(len(target_folder_list)):
                if folder_list[i].__contains__(target_folder_list[j]):
                    target_folder.append(folder_list[i])
                    break
        print(target_folder)

        # 设置 dataloader 长度
        if data_len < 0:
            self.length = 0
        else:
            self.length = data_len
        # print(self.length)

        # 把每一个文件夹，实例化成一个 dataset 对象
        self.dataset_ele = []
        for i in range(len(target_folder)):
            print('loading pic from path:{}'.format(target_folder_list[i].split('/')[-1]))
            self.dataset_ele.append(Ele_data(target_folder[i]))
            # print(dataset_ele[i].layer_info)
        self.windows_length = 1

    def get_item(self, index):
        if len(self.dataset_ele) != 1:
            index_folder = np.random.randint(0, len(self.dataset_ele ) -1)
        else:
            index_folder = 0
        print('index folder is :{}'.format(index_folder))

        # 随机选择一个 地层层段
        # print(self.dataset_ele[index_folder].layer_info.shape[0])
        index_layer_info = np.random.randint(0, self.dataset_ele[index_folder].layer_info.shape[0 ] -1)
        layer_info = self.dataset_ele[index_folder].layer_info[index_layer_info][-1]
        # print(index_layer_info)

        # print(self.dataset_ele[index_folder].layer_info[index_layer_info, :])

        # 根据层段 选出 层段的 起止深度信息
        dep_start, dep_end = self.dataset_ele[index_folder].layer_info[index_layer_info, 1:3]
        # print(dep_start, dep_end)

        # 如果 层 深度 接近1 则 直接选 层段中点
        if (dep_end - dep_start) < 1.001:
            depth_random = (dep_end + dep_start) / 2
        # 随机选着 深度
        else:
            depth_random = (np.random.random() * (dep_end - dep_start - 1)) + dep_start
        # print(depth_random)

        # 根据深度、层厚信息，获得图像信息
        depth_data_windows, pic_all, index_dep, depth_temp, thickness = \
            self.dataset_ele[index_folder].get_pic_from_depth_thickness(depth=depth_random)

        # print(index_dep, depth_temp, thickness)
        # print(pic_all[:, :5, :5])
        show_Pic([pic_all[1, :, : ], pic_all[2, :, : ]], pic_order='12', pic_str=[])

        return pic_all, layer_info


a = get_spilt_pic()
print(a[1])