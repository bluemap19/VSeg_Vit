from tqdm import trange

from src_ele.dir_operation import traverseFolder
from src_ele.file_operation import get_ele_data_from_path
from src_ele.pic_opeeration import get_pic_distribute
import numpy as np
import matplotlib.pyplot as plt

from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False


dist_l = 20


# 计算两曲线的DTW面积
def dtw_distance(s1, s2):
    """
    计算两个序列的DTW距离
    :param s1: 序列1
    :param s2: 序列2
    :return: DTW距离
    """
    n, m = len(s1), len(s2)
    dtw = np.zeros((n+1, m+1))
    for i in range(1, n+1):
        dtw[i][0] = float('inf')
    for i in range(1, m+1):
        dtw[0][i] = float('inf')
    dtw[0][0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(s1[i-1] - s2[j-1])
            dtw[i][j] = cost + min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])
    return dtw[n][m]

# 获取文件夹内 图像的 所有图片的 像素分布
def get_folder_pic_dist(path_t):
    paths = traverseFolder(path_t)
    # print(paths)

    stat_dist_list = []
    dyna_dist_list = []
    for i in trange(len(paths)):
        # print(i)
        if paths[i].__contains__('stat'):
            file_stat_t, _ = get_ele_data_from_path(paths[i])
            # print(file_stat_t.shape)
            pic_dist_t = get_pic_distribute(file_stat_t, dist_length=dist_l)
            stat_dist_list.append(pic_dist_t)
        else:
            file_dyna_t, _ = get_ele_data_from_path(paths[i])
            # print(file_dyna_t.shape)
            pic_dist_t = get_pic_distribute(file_dyna_t, dist_length=dist_l)
            dyna_dist_list.append(pic_dist_t)

    return np.array(stat_dist_list), np.array(dyna_dist_list)

# 计算 图像像素分布 list 的 平均值及 s2
def cal_dist_feature(dist_feature_list):
    dist_n = []
    s2 = []
    for j in range(dist_feature_list.shape[1]):
        dist_n.append(np.mean(dist_feature_list[:, j]))

    # print(dist_n)
    dist_n = np.array(dist_n)
    # print(dist_n)
    # print(dist_n.shape)
    for i in range(dist_feature_list.shape[0]):
        # manhattan_distance = lambda dist_n, (dist_feature_list[i, :]): np.abs(dist_n - dist_feature_list[i, :])
        # alignmentOBE = dtw(dist_n, dist_feature_list[i, :])
        # plt.plot(alignmentOBE.index1, alignmentOBE.index2)
        # plt.show()
        # print()
        # print(alignmentOBE.__doc__, alignmentOBE.__dict__)
        # alignmentOBE.plot(type="twoway",offset=1)
        s2_t = dtw_distance(dist_n, dist_feature_list[i, :])
        s2.append(s2_t/len(dist_n))
        pass
    s2 = np.array(s2)

    return dist_n, s2

# 获取一口井动静态图像的 平均像素分布、各自的s2
def get_feature_dis(path_t):
    stat_dis, dyna_dis = get_folder_pic_dist(path_t)
    dist_stat, s2_stat = cal_dist_feature(stat_dis)
    dist_dyna, s2_dyna = cal_dist_feature(dyna_dis)
    return dist_dyna, dist_stat, s2_dyna, s2_stat

path_dry = r'D:\Data\target_stage3_small_p\train\0'
path_vugs = r'D:\Data\target_stage3_small_p\train\2'
path_fracture = r'D:\Data\target_stage3_small_p\train\1'
path_hole = r'D:\Data\target_stage3_small_p\train\4'
# path_all = [path_fracture, path_dry, path_vugs, path_hole]
path_all = [path_dry, path_vugs, path_fracture, path_hole]

# 设置 横坐标 list
a = []
# print(a)
for i in range(dist_l):
    bei = 256//dist_l
    a.append(i*bei)
y_ticks = np.array([0.05, 0.10, 0.15, 0.20])
# title_ax = ['裂缝型静态成像像素分布', '干层静态成像像素分布', '孔洞型静态成像像素分布', '洞穴型静态成像像素分布']
title_ax = ['干层静态成像像素分布', '孔洞型静态成像像素分布', '裂缝型静态成像像素分布', '洞穴型静态成像像素分布']

fig = plt.figure(figsize=(12, 10))
for i in range(len(path_all)):
    dist_dyna, dist_stat, s2_dyna, s2_stat = get_feature_dis(path_all[i])
    # 横坐标 倒过来
    dist_dyna = dist_dyna[::-1]
    dist_stat = dist_stat[::-1]

    if i == 3:
        ax1 = fig.add_subplot(2, 2, i + 1)
        ax1.set_title(title_ax[i])
        ax1.set_ylabel('占比')
        ax1.set_xlabel('像素分布')
        ax1.set_ylim((0, 1.0))
        ax1.bar(a, dist_stat, width=7)
        print(np.sum(s2_stat), np.sum(s2_dyna))
        break

    # 显示 动态测井 图像像素分布
    # ax1 = fig.add_subplot(2, 4, i*2+1)
    ax1 = fig.add_subplot(2, 2, i+1)

    # ax1.label_outer()
    ax1.set_title(title_ax[i])
    # ax1.legend('*(HFCF??哇塞的')

    ax1.set_ylabel('占比')
    ax1.set_xlabel('像素分布')
    # ax1.yticks(y_ticks)
    # ax1.bar_label('sdajnsdjnasdna')
    ax1.set_ylim((0, 0.2))
    # Axes.set_xbound               #### 设置 X 轴的下限值和上限值
    # Axes.set_ybound               #### 设置 Y 轴的下限值和上限值
    # ax1.set_xlim((0, 256))
    ax1.bar(a, dist_stat, width=7)

    # # 显示 静态测井 图像像素分布
    # ax2 = fig.add_subplot(2, 4, i*2+2)
    # ax2.set_ylabel('占比')
    # ax2.set_xlabel('像素分布')
    # ax2.bar(a, dist_dyna, width=5)
    print(np.sum(s2_stat), np.sum(s2_dyna))
    # break

plt.show()

