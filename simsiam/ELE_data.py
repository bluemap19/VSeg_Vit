import argparse
import copy
import math
import random
import cv2
import numpy as np
import pandas as pd
from src_ele.dir_operation import traverseFolder, check_and_make_dir, traverseFolder_folder
from src_ele.file_operation import get_ele_data_from_path
from src_ele.pic_opeeration import pic_scale_simple, WindowsDataZoomer, pic_enhence_random

# 设置 都有什么类型的 储层
config_layer_class = ['干层', '裂缝型', '孔洞型', '裂缝孔洞型-裂缝型', '洞穴型', '裂缝洞穴型', '裂缝孔洞型-孔洞型', '其他型']
config_layer_class_curve = ['干层', '裂缝型', '孔洞型', '裂缝孔洞型', '洞穴型']
# 最小层厚，这个用于检查，小于这个层厚的话，报错
MIN_LAYER_THICKNESS = 0.49
# MIN_LAYER用于划分层，小于这个层厚的图像，将直接被抠出来，当作一个
MIN_LAYER = 2.6
# 层厚范围，这个用于分隔，把大的层随即分割成小的层
Thicness_Range = [1, 2]
# 设置 层信息文件的 列名
cols_name = ['Layer_Info', 'Depth_Start', 'Depth_End', '类型']



def pic_rotate(pic, pixel_rotate_random):
    if (pixel_rotate_random >= pic.shape[-1]) | (pixel_rotate_random <= 0):
        print('rotate angle is error...')
        exit(0)

    pic_new = np.copy(pic)
    # print('sssssssssssssssssssssssssssssssss')
    # print(pic_new.shape)

    pic_width = pic.shape[-1]
    pic_new[:, :, :pic_width-pixel_rotate_random] = pic[:, :, pixel_rotate_random:]
    pic_new[:, :, pic_width-pixel_rotate_random:] = pic[:, :, :pixel_rotate_random]

    return pic_new


def check_make_save_path(path_out):
    if path_out == '':
        print('path_out is null')
        exit(0)
    else:
        for i in range(len(config_layer_class)):
            path_temp = path_out + '/' + str(i)
            check_and_make_dir(path_temp)

class Ele_data_img:
    img_static = np.array([])
    img_dyna = np.array([])
    depth_data = np.array([])
    layer_info = np.array([])
    pic_shape = []

    def __init__(self, path_folder,  pic_shape=[0.9, 0.9]):
        # 初始化 图片形状
        self.pic_shape = pic_shape
        self.pic_index = 0
        self.charter = path_folder.split('\\')[-1].split('/')[-1]
        print('current folder charter is :{}'.format(self.charter))

        # 遍历文件夹 内 所有文件
        file_list = traverseFolder(path=path_folder)
        # print(file_list)
        self.path_in = path_folder
        # print(file_list)
        for i in range(len(file_list)):
            # 静态图像赋值
            if file_list[i].__contains__('cif_stat_full') and file_list[i].endswith('txt'):
                self.img_static, self.depth_data = get_ele_data_from_path(file_list[i])
                # print(self.img_static.shape)
            # 动态图像赋值
            elif file_list[i].__contains__('cif_dyna_full') and file_list[i].endswith('txt'):
                self.img_dyna, self.depth_data = get_ele_data_from_path(file_list[i])
                # print('read from path successfully as shape:{}'.format(self.img_dyna.shape))
            # 分层信息 赋值
            elif file_list[i].__contains__('img_layer_class') and file_list[i].endswith('xlsx'):
                # data_train =
                print('layer info is exiting and loading......')
                self.layer_info = pd.read_excel(file_list[i])[cols_name].values
                # print(target)
                # print(type(target[0][0]), type(target[0][1]), type(target[0][2]), type(target[0][3]))

        # 判断 是否成功读取 原始电阻率图像 和 静态成像
        if (len(self.img_dyna)==0) | (len(self.img_static) == 0):
            print('img data is empty ,dyna is :{}, stat img is:{}'.format(self.img_dyna.shape, self.img_static.shape))
            exit(0)

        # print(self.depth_data.shape, self.depth_data)
        # 初始化 图像 深度 起始、终止 位置
        self.dep_start = self.depth_data[0, 0]
        self.dep_end = self.depth_data[-1, 0]

        # 设置文件的保存位置
        # self.path_out = args.DIR_unsupervised_out
        self.path_out = ''

        # 计算初始化 图像的 步长/分辨率
        self.LEV = (self.depth_data[-1, 0] - self.depth_data[0, 0]) / self.img_dyna.shape[0]

        # self.windows_step = int(args.windows_step/self.LEV)
        # self.windows_length = int(args.windows_length/self.LEV)
        self.windows_step = 200
        self.windows_length = 600

        # print(self.dep_start, self.dep_start_layer, self.dep_end, self.dep_end_layer)
        self.stat_use = True

        # 初始化 分层的 深度 起始、终止 位置
        self.dep_start_layer = -1
        self.dep_end_layer = -1
        if len(self.layer_info) == 0:
            print('layer info is null............')
        else:
            # print('ddddddddddddd:{}'.format(self.layer_info))
            self.dep_start_layer = self.layer_info[0, 1]
            self.dep_end_layer = self.layer_info[-1, 2]
            # 检查分层信息是否可靠
            self.check_layer_info()
            # 把分层的 字符串信息 转化成 数字信息
            self.replace_layer_str()

        # if len(self.path_out) > 0:
        #     check_make_save_path(self.path_out)


    def get_index(self, depth):
        index_cal = int((depth-self.dep_start)/self.LEV)

        index_start = max(index_cal - 10, 0)

        index_tar = 0
        for i in range(self.img_dyna.shape[0]-index_start-2):
            if self.depth_data[i+index_start][0] >= depth:
                index_tar = i+index_start
                break

        return index_tar


    # 检查分层信息
    def check_layer_info(self):
        for i in range(self.layer_info.shape[0]):
            # 层厚不够，报错
            if (self.layer_info[i][2] - self.layer_info[i][1]) < MIN_LAYER_THICKNESS:
                print('layer is too thick, start&end depth is:{},{}'.format(self.layer_info[i][1], self.layer_info[i][2]))
                exit(0)
            # 底深大于顶深 报错
            if self.layer_info[i][1] > self.layer_info[i][2]:
                print('start dep is bigger than end dep:{},{}'.format(self.layer_info[i][1], self.layer_info[i][2]))
                exit(0)
            # 如果 没有该层的类型信息 报错
            if self.layer_info[i][3] not in config_layer_class:
                print('layer info not in config_layer_class：{}'.format(self.layer_info))
                exit(0)
            if i != 0:
                # 顶深 小于 上一层底深 报错
                if self.layer_info[i][1] < self.layer_info[i - 1][2]:
                    print('start dep is smaller than last dep:{}, {}'.format(self.layer_info[i][1], self.layer_info[i - 1][2]))
                    exit(0)

        # 如果 层位信息的开始深度小于图像开始深度 或 层位信息的结束深度大于图像结束深度 则 报错
        if (self.dep_start_layer < self.dep_start) | (self.dep_end_layer > self.dep_end):
            print('layer info is something wrong,layer from {} to {}, depth from {} to {}'.format(
                self.dep_start_layer, self.dep_end_layer, self.dep_start, self.dep_end))
            exit(0)
        print('check file successfully')


    # 把 文字版的 储层类型 换成 数字版 方便计算
    def replace_layer_str(self):
        for i in range(self.layer_info.shape[0]):
            self.layer_info[i][-1] = config_layer_class.index(self.layer_info[i][-1])


    # 根据深度获得随机图片
    def get_pic_from_depth(self, depth=5999, thickness = (random.random() * (Thicness_Range[1] - Thicness_Range[0])) + Thicness_Range[0],
                           save=False, DIR=''):
        if (depth < self.dep_start) | (depth > self.dep_end):
            print('depth error,search depth:{}, srart&end depth:{},{}'.format(depth, self.dep_start, self.dep_end))
            exit(0)
        # 根据深度 获得该从哪一个 索引处 开始扣取图片
        index_dep = self.get_index(depth)

        # 根据图片的长度信息，获得窗口的像素个数
        pixel_windows = int(thickness/self.LEV)

        # # 如果 图片边长为 百分比小数形式，则乘上真正边长
        # if self.pic_shape[0] <= 1:
        #     self.pic_shape = [int(pixel_windows*self.pic_shape[0]), int(self.img_dyna.shape[1]*self.pic_shape[1])]

        index_start = max(index_dep - pixel_windows//2, 0)
        index_end = index_start + pixel_windows

        # print(index_start, index_end, pixel_windows, index_dep)
        # print(self.img_dyna[index_start:index_end, :].shape)

        # 根据 深度的索引 获得 对应的 窗口原始电阻率矩阵信息 和 窗口静态矩阵信息
        pic_dyna_windows = self.img_dyna[index_start:index_end, :]
        pic_static_windows = self.img_static[index_start:index_end, :]
        depth_data_windows = self.depth_data[index_start:index_end, :]
        # print('pic org shape is :{}'.format(pic_dyna_windows.shape))

        if save:
            # print(self.charter, 0, depth_data_windows[0, 0], depth_data_windows[-1, 0])
            # print(DIR)
            path_temp = DIR + '\\{}_{}_{:.4f}_{:.4f}_dyna.png'.format(self.charter, 0, depth_data_windows[0, 0], depth_data_windows[-1, 0])
            print(path_temp)
            cv2.imwrite(path_temp, pic_dyna_windows)
            if self.stat_use:
                path_temp = path_temp.replace('dyna', 'stat')
                cv2.imwrite(path_temp, pic_static_windows)

        pic_dyna_windows = cv2.resize(pic_dyna_windows.astype(np.float32), (224, 224)) / 256
        pic_static_windows = cv2.resize(pic_static_windows.astype(np.float32), (224, 224)) / 256

        return pic_dyna_windows, pic_static_windows, depth_data_windows
        # # 简单的数据缩放
        # pic_dyna_windows_scale = pic_scale_simple(pic=pic_dyna_windows, pic_shape=self.pic_shape)
        # pic_static_windows_scale = pic_scale_simple(pic=pic_static_windows, pic_shape=self.pic_shape)
        # # 对 原始数据矩阵进行 动态增强
        # pic_dynamic_windows_scale_temp, _, _ = WindowsDataZoomer(pic_dyna_windows_scale)
        # pic_dynamic_windows_scale = pic_enhence_random(pic_dynamic_windows_scale_temp)
        #
        # # 数据合并 把三个数据 合并成一个数据 原始矩阵，静态矩阵，动态矩阵
        # pic_all = np.zeros((3, self.pic_shape[0], self.pic_shape[1]))
        #
        # pic_all[0][:, :] = pic_dyna_windows_scale
        # pic_all[1][:, :] = pic_static_windows_scale
        # pic_all[2][:, :] = pic_dynamic_windows_scale
        #
        # # # 随机的 图形 绕井周旋转
        # # pixel_rotate_random = int(random.random() * self.pic_shape[1] * 0.98) + 1
        # # pic_all = pic_rotate(pic_all, pixel_rotate_random)
        #
        # # pic_scale(input, windows_shape=3, center_ratio=0.5, x_size=100.0, y_size=100.0, ratio_top=0.1):
        # # pic_enhence_random(input, windows_shape=3, ratio_top=0.2, ratio_migration=0.6, random_times=3):
        #
        # return depth_data_windows, pic_all, index_dep, self.depth_data[index_dep, 0], thickness



    # 为stage1 生成 无监督训练数据集
    def pic_spilt_directlly_for_unsupervised_learning(self, path=''):
        if path != '':
            self.path_out = path
            check_and_make_dir(self.path_out)

        # 先计算要进行多少次分割，再对结果进行保存
        pic_num = ((self.img_dyna.shape[0] - self.windows_length) // self.windows_step) + 1
        for j in range(pic_num):
            self.pic_index += 1
            index_start_temp = j * self.windows_step

            if j%2 == 0:
                windows_length_temp = 250
            else:
                windows_length_temp = 500
            index_end_temp = index_start_temp + windows_length_temp
            # index_end_temp = min(j * self.windows_step + self.windows_length + np.random.randint(-30, 60), self.img_dyna.shape[0]-1)

            path_temp = '{}/{}_{}_{:.4f}_{:.4f}_dyna.png'.format(self.path_out,
                                                                    self.path_in.split('/')[-1].split('\\')[-1],
                                                                    self.pic_index,
                                                                    self.depth_data[index_start_temp][0],
                                                                    self.depth_data[index_end_temp][0])
            print('make unsupervised layer:{},{}'.format(j, path_temp))

            img_temp = self.img_dyna[index_start_temp:index_end_temp, :]
            cv2.imwrite(path_temp, img_temp)
            if self.stat_use:
                path_temp = path_temp.replace('dyna', 'stat')
                img_temp = self.img_static[index_start_temp:index_end_temp, :]
                cv2.imwrite(path_temp, img_temp)

    # 用于stage3
    # 根据 分层信息 将图片进行小窗口的，并存储在不同文件夹内
    def pic_auto_spilt_by_layer_to_small_layer(self, path=''):
        if path != '':
            self.path_out = path
            check_make_save_path(self.path_out)

        for i in range(self.layer_info.shape[0]):
            print(self.layer_info[i], self.get_index(self.layer_info[i][1]), self.get_index(self.layer_info[i][2]))

            index_start, index_end = self.get_index(self.layer_info[i][1]), self.get_index(self.layer_info[i][2])
            # print(float(self.layer_info[i][2]) - float(self.layer_info[i][2]), self.layer_info[i][2], self.layer_info[i][2])

            # 判断是否小于最小层厚，如果是的，直接把这一层分成单独的，直接保存
            # if ((float(self.layer_info[i][2]) - float(self.layer_info[i][1])) <= MIN_LAYER):
            if ((float(self.layer_info[i][2]) - float(self.layer_info[i][1])) <= 0.6):
                # 直接根据分类类别 对结果进行保存
                if ((float(self.layer_info[i][2]) - float(self.layer_info[i][1])) <= 0.5):
                    pass
                self.pic_index += 1
                path_temp = '{}/{}/{}_{}_{:.4f}_{:.4f}_dyna.png'.format(self.path_out, str(self.layer_info[i][3]), self.path_in.split('/')[-1].split('\\')[-1],
                                                       self.pic_index, self.layer_info[i][1], self.layer_info[i][2])
                print('stage3 small full layer:'+path_temp)
                img_temp = self.img_dyna[index_start:index_end, :]
                cv2.imwrite(path_temp, img_temp)
                # print(self.stat_use, type(self.stat_use))
                if self.stat_use:
                    path_temp_stat = path_temp.replace('dyna', 'stat')
                    # print(path_temp_stat)
                    img_temp_stat = self.img_static[index_start:index_end, :]
                    cv2.imwrite(path_temp_stat, img_temp_stat)
            else:
                # 层厚太大，要对其进行分割
                # 先计算要进行多少次分割，再对结果进行保存
                pic_num = (((index_end-index_start) - self.windows_length) // self.windows_step) + 1
                for j in range(pic_num):
                    self.pic_index += 1
                    index_start_temp = index_start + j*self.windows_step

                    # windows_length_temp = int((np.random.random()*(Thicness_Range[1]-Thicness_Range[0])+Thicness_Range[0])/self.LEV)
                    windows_length_temp = np.random.randint(250, 270)
                    index_end_temp = index_start_temp + windows_length_temp
                    # index_end_temp = min(j * self.windows_step + self.windows_length + np.random.randint(-30, 60), self.img_dyna.shape[0]-1)

                    path_temp = '{}/{}/{}_{}_{:.4f}_{:.4f}_dyna.png'.format(self.path_out, str(self.layer_info[i][3]),
                                                                   self.path_in.split('/')[-1].split('\\')[-1],
                                                                   self.pic_index, self.depth_data[index_start_temp][0],
                                                                   self.depth_data[index_end_temp][0])
                    print('stage3 large split layer:{},{}'.format(j, path_temp))
                    img_temp = self.img_dyna[index_start_temp:index_end_temp, :]
                    cv2.imwrite(path_temp, img_temp)
                    # print(self.stat_use, type(self.stat_use))
                    if self.stat_use:
                        path_temp_stat = path_temp.replace('dyna', 'stat')
                        # print(path_temp_stat)
                        img_temp_temp = self.img_static[index_start_temp:index_end_temp, :]
                        cv2.imwrite(path_temp_stat, img_temp_temp)


    # 适用于stage3
    # 根据 分层信息 将图片进行不同类型的划分,并不切割成小的文件形式，直接将整个大的层进行保存，存储在不同文件夹内
    def pic_auto_spilt_by_layer_to_fulllayer(self, path=''):
        if path != '':
            self.path_out = path
            check_and_make_dir(self.path_out)
            for i in range(len(config_layer_class)):
                path_temp = self.path_out + '/' + str(i)
                check_and_make_dir(path_temp)
        for i in range(self.layer_info.shape[0]):
            self.pic_index += 1
            print(self.layer_info[i], self.get_index(self.layer_info[i][1]), self.get_index(self.layer_info[i][2]))
            # 根据深度信息，计算层的起止index
            index_start, index_end = self.get_index(self.layer_info[i][1]), self.get_index(self.layer_info[i][2])
            # print(float(self.layer_info[i][2]) - float(self.layer_info[i][2]), self.layer_info[i][2], self.layer_info[i][2])

            # 直接根据分类类别 对结果进行保存
            path_temp = '{}/{}/{}_{}_{:.4f}_{:.4f}_{}.png'.format(self.path_out, str(self.layer_info[i][3]),
                                                                    self.path_in.split('/')[-1].split('\\')[-1],
                                                                    self.pic_index, self.layer_info[i][1],
                                                                    self.layer_info[i][2], 'dyna')
            print('make layer:' + path_temp)
            img_temp = self.img_dyna[index_start:index_end, :]
            cv2.imwrite(path_temp, img_temp)
            if self.stat_use:
                path_temp_vice = path_temp.replace('dyna', 'stat')
                img_temp_vice = self.img_static[index_start:index_end, :]
                cv2.imwrite(path_temp_vice, img_temp_vice)


# def layer_combine_process(target_Inf):
#     jump_num = 0
#
#     processed_target = []
#     start_dep = 0
#     end_dep = 0
#     for i in range(target_Inf.shape[0]):
#         if jump_num != 0:
#             jump_num -= 1
#             # print('jump num:{}'.format(jump_num))
#             continue
#         else:
#             start_dep = target_Inf[i][0]
#             end_dep = target_Inf[i][1]
#             inf_type = int(target_Inf[i][2])
#             while ((i + jump_num) < (target_Inf.shape[0]-1)):
#                 jump_num += 1
#                 end_dep_next = target_Inf[i+jump_num][1]
#                 inf_type_next = int(target_Inf[i+jump_num][2])
#                 if inf_type_next == inf_type:
#                     end_dep = end_dep_next
#                     continue
#                 else:
#                     jump_num -= 1
#                     break
#             item = np.array([start_dep, end_dep, inf_type])
#             processed_target.append(item)
#
#     # print(len(processed_target))
#     return np.array(processed_target)

# # 把二维的地层信息合并成地层表格信息
# def combine_class_inf_to_table(result_class):
#     jump_num = 1
#     processed_target = []
#     start_dep = 0
#     end_dep = 0
#     for i in range(result_class.shape[0]-1):
#         if jump_num != 1:
#             jump_num -= 1
#             continue
#         else:
#             start_dep = result_class[i][0]
#             end_dep = result_class[i+1][0]
#             inf_type = int(result_class[i][-1])
#             while (i+jump_num < result_class.shape[0]-1):
#                 end_dep = result_class[i+jump_num][0]
#                 inf_type_end = int(result_class[i+jump_num][-1])
#                 if inf_type_end == inf_type:
#                     jump_num += 1
#                     continue
#                 else:
#                     break
#
#             item = np.array([start_dep, end_dep, inf_type])
#             processed_target.append(item)
#     return np.array(processed_target)




np.set_printoptions (suppress=True)
# Normal_char = ['DEPTH', 'CAL',  'GR',   'SP',   'RD',   'RS',   'CN',   'DEN',  'DT24']
NORMAL_RANGE_CONFIG = [ 3,      0,      -98,    0,      0,      -1,      0,     0]
Normal_error_replace =[ 999999, 999999, 999999, 999999, 999999, 999999, 999999, 999999]
Error_replace = 999999
NORMAL_CHAR = ['DEPTH', 'CAL',  'GR',   'SP',   'RD',   'RS',   'CN',   'DEN',  'DT24']
def normal_log_clean(n_logging_o=np.random.rand(5, 5)):
    # 数据清洗，只有SP测井值可以为负数，但是不能是-999以下的

    for i in range(n_logging_o.shape[0]):
        for j in range(n_logging_o.shape[1]):
            if n_logging_o[i][j] < NORMAL_RANGE_CONFIG[j]:
                n_logging_o[i][j] = Error_replace
    return n_logging_o


# # 该方法存在的问题很大，极小值附近，存在一大帮极大值，会直接使结果爆炸
# # 获取极小值附近一段的平均值，当作，层段的相对小值
# def get_relative_small_value(log_data, pix_min_windows_len=50):
#     v_min_tgrsv = np.min(log_data)
#     log_data_temp = log_data.tolist()
#     index_min_tgrsv = log_data_temp.index(v_min_tgrsv)
#     # print(index_min_tgrsv)
#     index_start_tgrsv = index_min_tgrsv - pix_min_windows_len//2
#     index_end_tgrsv = index_start_tgrsv + pix_min_windows_len
#     v_min_mean = np.mean(log_data[index_start_tgrsv:index_end_tgrsv])+0.0001
#     return v_min_mean
# 获取该层段，最小10%的平均值，当作，层段的相对小值
def get_relative_small_value_by_sort(log_data, pix_min_windows_len=50):
    log_data_N = []
    for i in log_data:
        if i < Error_replace:
            log_data_N.append(i)
    log_data_N = np.array(log_data_N)

    # print('before shape:{}, drop shape:{}'.format(log_data.shape, log_data_N.shape))
    if len(log_data_N) < 10:
        return Error_replace, Error_replace
    else:
        v_max_mean = np.mean(np.sort(log_data_N)[-pix_min_windows_len:])
        v_min_mean = np.mean(np.sort(log_data_N)[:pix_min_windows_len])
        return v_min_mean, v_max_mean

# 曲线归一化
def normal_log_regularization(n_logging_o=np.random.rand(5, 5), R_cols=[3, 4], windows_length=500, pix_min_ratio=0.2):
    n_logging_N = copy.deepcopy(n_logging_o)
    # 数据清洗，把不合规的数据，统一替换为
    n_logging_N = normal_log_clean(n_logging_N)

    # 对电阻率进行对数预处理
    for i in range(n_logging_N.shape[0]):
        for j in R_cols:
            if n_logging_N[i][j] < 999999:
                # print('对数处理：')
                # print(n_logging_N[i][j], math.log(n_logging_N[i][j]))
                n_logging_N[i][j] = math.log(n_logging_N[i][j]+1, 10)
            else:
                n_logging_N[i][j] = n_logging_N[i][j]


    # n_logging = np.zeros_like(n_logging_N)
    n_logging = np.zeros((n_logging_N.shape[0], n_logging_N.shape[1]+1))
    for j in range(n_logging_N.shape[1]):
        # 井径，不做什么处理
        if j == 0:
            for i in range(n_logging_N.shape[0]):
                n_logging[i][j] = n_logging_N[i][j]
            # pass
        # 中子，不做处理
        elif j==5:
            for i in range(n_logging.shape[0]):
                n_logging[i][j] = n_logging_N[i][j]
        # 浅电阻率 不仅 保留原始数值不变，还保留了 局部地区的相对高低
        elif j==4:
            for i in range(n_logging.shape[0]):
                n_logging[i][j] = n_logging_N[i][j]

                index_start_tnlr = max(0, i-windows_length//2)
                index_end_tnlr = min(index_start_tnlr + windows_length, n_logging_o.shape[0]-1)
                index_start_tnlr = index_end_tnlr - windows_length
                V_min_mean_layer, V_max_mean_layer = get_relative_small_value_by_sort(n_logging_N[index_start_tnlr:index_end_tnlr, j].ravel(), pix_min_windows_len=int(windows_length*pix_min_ratio))
                n_logging[i][-1] = (n_logging_N[i][j] - V_min_mean_layer)/(V_min_mean_layer+0.001)
        # 深电阻率，单独处理
        elif j == 3:
            for i in range(n_logging.shape[0]):
                if (n_logging_N[i][j]<Error_replace)&(n_logging_N[i][j+1]<Error_replace):
                    n_logging[i][j] = n_logging_N[i][j]-n_logging_N[i][j+1]
                else:
                    n_logging[i][j] = n_logging_N[i][j]
        # 其他信息，一律进行局部特征提取
        else:
            for i in range(n_logging.shape[0]):
                if n_logging_N[i][j] >= 999999:
                    n_logging[i][j] = n_logging_N[i][j]
                else:
                    index_start_tnlr = max(0, i-windows_length//2)
                    index_end_tnlr = min(index_start_tnlr + windows_length, n_logging_o.shape[0]-1)
                    index_start_tnlr = index_end_tnlr - windows_length
                    # print(index_start_tnlr-index_end_tnlr, windows_length//2)
                    V_min_mean_layer, V_max_mean_layer = get_relative_small_value_by_sort(n_logging_N[index_start_tnlr:index_end_tnlr, j].ravel(), pix_min_windows_len=int(windows_length*pix_min_ratio))
                    n_logging[i][j] = (n_logging_N[i][j] - V_min_mean_layer)/(V_min_mean_layer+0.001)
                    # print('current value:{}, small layer mean:{}, change to :{}'.format(n_logging_o[i][j], V_min_mean_layer, n_logging[i][j]))
    return n_logging

# 这个只能对连续深度的数据起作用，不是连续数据的话，不能用这个
# 根据深度，获取指定数据的 曲线信息 并不返回深度信息
def get_data_from_depth(dep_temp, curve):
    dep_start = curve[0][0]
    dep_end = curve[-1][0]

    if dep_temp < (dep_start-0.01):
        print('error dep:'.format(dep_temp))
        exit(0)

    index_start = max(int((dep_temp-dep_start)/(dep_end-dep_start)*curve.shape[0]) - 10, 0)

    index_loc = 0
    for i in range(curve.shape[0] - index_start):
        if curve[index_start+i][0] > dep_start:
            index_loc = index_start+i
            break

    if index_loc==0:
        return curve[index_loc, 1:]
    elif index_loc==curve.shape[0]-1:
        return curve[index_loc, 1:]
    else:
        return (curve[index_loc, 1:]+curve[index_loc-1, 1:])/2

# 两条测井数据合并函数，每个测井数据的第一列必须是深度信息
def two_curve_info_combine(curve1, curve2, LEV=0.075):
    ###
    # 两个曲线数据合并, 返回的数据里面有新的深度数据
    # ###
    depth_1_start = curve1[0][0]
    depth_2_start = curve2[0][0]
    depth_1_end = curve1[-1][0]
    depth_2_end = curve2[-1][0]

    dep_start = max(depth_1_start, depth_2_start)
    dep_end = min(depth_1_end, depth_2_end)

    charter_full = []
    num_length = int((dep_end-dep_start)/LEV)
    for i in range(num_length):
        dep_temp = dep_start + i*LEV
        c1_temp = get_data_from_depth(dep_temp, curve1)
        c2_temp = get_data_from_depth(dep_temp, curve2)

        char_temp = np.append(c1_temp, c2_temp, axis=None)
        char_temp = np.append(dep_temp, char_temp)

        charter_full.append(char_temp)

    return np.array(charter_full)


# 获取指定曲线 指定比例 指定范围内的 最大值、最小值
def get_max_value_by_ratio(curve=np.array([]), ratio_c=0.2, range_c=[-1, 9999]):
    max_list = []
    min_list = []

    d_t = curve.ravel()

    num_ratio = int(d_t.shape[0] * ratio_c)
    print(d_t.shape)
    for i in range(d_t.shape[0]):
        if (d_t[i]<range_c[0]) | (d_t[i]>range_c[1]):
            continue

        if len(max_list) < num_ratio:
            # 初始化 最大列表、最小列表
            max_list.append(d_t[i])
            min_list.append(d_t[i])

        else:
            max_from_min_list = max(min_list)
            min_from_max_list = min(max_list)

            if d_t[i] > min_from_max_list:
                index_min = max_list.index(min_from_max_list)
                max_list[index_min] = d_t[i]

            if d_t[i] < max_from_min_list:
                index_max = min_list.index(max_from_min_list)
                min_list[index_max] = d_t[i]

    max_list = np.array(max_list)
    min_list = np.array(min_list)
    # print((np.sort(min_list), np.sort(max_list)))
    return np.mean(max_list), np.mean(min_list)

# a = np.array([1.2, 2.3, 3.4, 4.5, 5.4,
#               6.3, 7.2, 8.1, 9.0, 0.2,
#               1.4, 2.3, 3.2, 4.2, 5.3,
#               6.2, 11.7, 12.0, 0.3, 2.1,
#               7.2, 3.1, 6.3, 5.9, 5.4,
#               7.5, 2.1, 3.1, 3.6, 6.3,
#               5.1, 3.2, 6.9, 7.8, 9.9, 1.2])
# # (array([1.2, 2.1, 0.3, 2.1, 1.2, 1.4, 0.2]), array([ 8.1,  9. , 11.7,  7.8, 12. ,  7.5,  9.9]))
# # (array([1.2, 2.3, 1.2, 2.1, 2.1, 2.3, 1.4]), array([8.1, 6.9, 7.8, 7.2, 7.5, 6.3, 7.2]))
# print(get_max_value_by_ratio(a))
# print(get_max_value_by_ratio(a, range_c=[0.5, 8.5]))


class Ele_data_normal:

    def __init__(self, path_folder, VIEW_LENGTH = 30, binary_pred=True):
        self.charter = path_folder.split('\\')[-1].split('/')[-1]
        print('current folder charter is :{}'.format(self.charter))

        # 遍历文件夹 内 所有文件
        file_list = traverseFolder(path=path_folder)
        # print(file_list)
        self.path_in = path_folder
        self.LEV_normal = 0
        self.LEV_ele_img = 0
        self.layer_list = np.array([])
        # print(file_list)
        for i in range(len(file_list)):
            # 常规测井曲线载入
            if file_list[i].__contains__('Normal') and file_list[i].endswith('xlsx'):
                self.curve_Normal = pd.read_excel(file_list[i])[NORMAL_CHAR].values
                # print('self.curve_Normal shape:{}'.format(self.curve_Normal.shape))
                self.Depth_curve = self.curve_Normal[:, 0].reshape((-1, 1))
                self.curve_Normal = self.curve_Normal[:, 1:]
                # 计算 常规九条测井数据的分辨率
                self.LEV_normal = (self.Depth_curve[-1, 0] - self.Depth_curve[0, 0]) / self.Depth_curve.shape[0]
                # print(self.img_static.shape)

            # 图像特征赋值
            elif file_list[i].__contains__('pred_answer') and file_list[i].endswith('txt'):
                self.char_img, self.Depth_img = get_ele_data_from_path(file_list[i])
                # print('self.char_img shape:{}'.format(self.curve_Normal.shape))
                # print('read from path successfully as shape:{}'.format(self.img_dyna.shape))
                # 计算图像特征数据的分辨率
                self.LEV_ele_img = (self.Depth_img[-1, 0] - self.Depth_img[0, 0]) / self.Depth_img.shape[0]

            # 分层信息 赋值
            elif file_list[i].__contains__('layer_info') and file_list[i].endswith('xlsx'):
                # print('layer info is exiting and loading......')
                self.layer_info = pd.read_excel(file_list[i])[cols_name].values
                self.replace_layer_str()

        # 判断 是否成功读取 常规九条
        if (len(self.curve_Normal)==0):
            print('curve data is empty ,curve_Normal is :{}, char_img is:{}'.format(self.curve_Normal.shape, self.char_img.shape))
            exit(0)


        # 判断 是否成功读取 成像特征信息，如果没有，赋值为倆列0
        if  (len(self.char_img) == 0):
            print('no img information ,reset the img info as 0')
            self.char_img = np.zeros((self.curve_Normal.shape[0], 4))
            self.Depth_img = self.Depth_curve


        if self.LEV_normal < 0.001:
            print('error self.LEV_normal:{}'.format(self.LEV_normal))
            exit(0)


        # print(int(VIEW_LENGTH/self.LEV_normal), self.LEV_normal)
        # 常规九条曲线的特征工程
        self.curve_Normalize = normal_log_regularization(self.curve_Normal, windows_length=int(VIEW_LENGTH/self.LEV_normal), pix_min_ratio=0.2)


        # 常规九条归一化后的测井数据 加上深度信息
        self.curve_Normalize_with_depth = np.hstack((self.Depth_curve, self.curve_Normalize))
        # curve_Normalize_with_depth[np.where((curve_Normalize_with_depth == Error_replace))] = -9999
        # np.savetxt(path_folder + '\\log_norrmalize_{}.txt'.format(self.charter),
        #            curve_Normalize_with_depth,
        #            delimiter='\t',
        #            header='#{}\t{}_reg\t{}_reg\t{}_reg\t{}_reg\t{}_reg\t{}_reg\t{}_reg\t{}_reg'.format(NORMAL_CHAR[0], NORMAL_CHAR[1],
        #            NORMAL_CHAR[2], NORMAL_CHAR[3], NORMAL_CHAR[4], NORMAL_CHAR[5], NORMAL_CHAR[6], NORMAL_CHAR[7], NORMAL_CHAR[8]),
        #            fmt='%.4f', comments='')


        # 把 经过特征工程的 常规九条曲线 与 图像特征曲线 合并到一起
        self.Curve_final = two_curve_info_combine(self.curve_Normalize_with_depth, np.hstack((self.Depth_img, self.char_img)))
        # print(self.Curve_final.shape)


        # # 初始化 线性数据 深度 起始、终止 位置
        self.dep_start = self.Curve_final[0, 0]
        self.dep_end = self.Curve_final[-1, 0]


        # # 计算初始化 整体数据的 步长/分辨率
        self.LEV = (self.Curve_final[-1, 0] - self.Curve_final[0, 0]) / self.Curve_final.shape[0]


        # 把预测结果的 table 转换成 list
        self.layer_list = self.trans_layer_class_table_2_class_list()


        if binary_pred:
            print('set pred answer as binary')
            self.layer_list = self.set_pred_as_binary()


        self.train_data = self.combine_layer_class_and_curve()
        self.train_data_without_img_char = self.combine_layer_class_and_curve_without_img_char()

    def get_all_curve(self):
        return self.Curve_final

    def get_processed_curve_by_dep(self, dep):
        return get_data_from_depth(dep, self.Curve_final)

    # 把 文字版的 储层类型 换成 数字版 方便计算
    def replace_layer_str(self):
        for i in range(self.layer_info.shape[0]):
            self.layer_info[i][-1] = config_layer_class_curve.index(self.layer_info[i][-1])

    def get_layer_class_and_curve_info_by_dep(self):
        inf_layer_class_normal_curve = []
        for i in range(self.layer_info.shape[0]):
            # print(self.layer_info[i])
            num_split = int((self.layer_info[i][2] - self.layer_info[i][1])/self.LEV)
            for j in range(num_split):
                dep_temp = self.layer_info[i][1] + j*self.LEV
                if (dep_temp>self.dep_start) & (dep_temp<self.dep_end):
                    c_temp = get_data_from_depth(dep_temp, self.Curve_final)
                    c_temp = np.append(dep_temp, c_temp)
                    c_temp = np.append(c_temp, self.layer_info[i][-1])
                    inf_layer_class_normal_curve.append(c_temp)

        return np.array(inf_layer_class_normal_curve)

    def trans_layer_class_table_2_class_list(self):
        layer_list = []
        for i in range(self.layer_info.shape[0]):
            dep_start_t = self.layer_info[i][1]
            dep_end_t = self.layer_info[i][2]
            # 根据深度信息，计算该层有多少的数据点
            pix_num = int((dep_end_t-dep_start_t)/self.LEV_normal)
            class_t = self.layer_info[i][3]

            for j in range(pix_num):
                dep_t = dep_start_t + j * self.LEV_normal
                layer_list.append([dep_t, class_t])

        return np.array(layer_list)

    def combine_layer_class_and_curve(self):
        info_all = []
        for i in range(self.layer_list.shape[0]):
            dep_t = self.layer_list[i][0]
            if (dep_t < self.dep_start) | (dep_t > self.dep_end):
                continue
            else:
                class_type = self.layer_list[i][1]
                # print(dep_t)
                info_curve_t = get_data_from_depth(dep_t, self.Curve_final)
                v_t = np.append(dep_t, info_curve_t)
                v_t = np.append(v_t, class_type)
                info_all.append(v_t)

        return np.array(info_all)

    def combine_layer_class_and_curve_without_img_char(self):
        info_all = []
        for i in range(self.layer_list.shape[0]):
            dep_t = self.layer_list[i][0]
            if (dep_t < self.dep_start) | (dep_t > self.dep_end):
                continue
            else:
                class_type = self.layer_list[i][1]
                # print(dep_t)
                info_curve_t = get_data_from_depth(dep_t, self.curve_Normalize_with_depth)
                v_t = np.append(dep_t, info_curve_t)
                v_t = np.append(v_t, class_type)
                info_all.append(v_t)

        return np.array(info_all)

    def set_pred_as_binary(self):
        lay_n = np.zeros_like(self.layer_list)
        for i in range(self.layer_list.shape[0]):
            lay_n[i][0] = self.layer_list[i][0]
            if self.layer_list[i][-1] > 0.5:
                lay_n[i][-1] = 1
            else:
                lay_n[i][-1] = 0
        return lay_n


# a = Ele_data_normal(path_folder=r'D:\Data\target5\LN11-5')
# # print(a.get_layer_class_and_curve_info_by_dep().shape)
# # # a = Ele_data(r'D:\1111\target4\zg111', args)
# # # a.pic_spilt_directlly_for_unsupervised_learning(args.DIR_unsupervised_out)
# # a = Ele_data(r'D:\1111\target4\zg112', args)
# # a.pic_spilt_directlly_for_unsupervised_learning(args.DIR_unsupervised_out)
# print(a.layer_info)
# print('Normalized curve shape is :{}'.format(a.curve_Normalize.shape))
# print('Img char shape is :{}'.format(a.char_img.shape))
# print('normal curve LEV is :{}, img charter LEV is :{}'.format(a.LEV_normal, a.LEV_ele_img))
# print('final train data shape is :{}'.format(a.train_data.shape))
# print('final train data without img charter shape is :{}'.format(a.train_data_without_img_char.shape))
# print(a.train_data[:2, :])


def get_train_data():
    proj_folder = r'D:\Data\target5'
    list_folder = traverseFolder_folder(proj_folder)
    skip_folder = ['LG7-11']
    info_all = []
    for i in list_folder:
        # print(i)
        skip_flag = False
        for j in skip_folder:
            if i.__contains__(j):
                skip_flag = True
                break
        if skip_flag:
            print('skip current folder:{}'.format(i))
            continue
        else:
            pass
        a = Ele_data_normal(path_folder=i, VIEW_LENGTH=30, binary_pred=True)

        print('Normalized curve shape is :{}'.format(a.curve_Normalize.shape))
        print('Img char shape is :{}'.format(a.char_img.shape))
        print('normal curve LEV is :{:.3f}, img charter LEV is :{:.2f}'.format(a.LEV_normal, a.LEV_ele_img))
        print('final train data shape is :{}'.format(a.train_data.shape))
        print('final train data without img charter shape is :{}'.format(a.train_data_without_img_char.shape))

        print('\n')
        if len(info_all)==0:
            info_all = a.train_data_without_img_char
        else:
            info_all = np.vstack((info_all, a.train_data_without_img_char))

    return info_all

# a = get_train_data()
# print(a.shape)
# print(a[2000:2002, :])