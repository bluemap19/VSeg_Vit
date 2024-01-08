import argparse
import cv2
import numpy as np
from tqdm import trange

from simsiam.ELE_data import Ele_data_img
from src_ele.dir_operation import traverseFolder_folder, traverseFolder, check_and_make_dir
from src_ele.file_operation import get_ele_data_from_path

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# metavar - 在 usage 说明中的参数名称，对于必选参数默认就是参数名称，对于可选参数默认是全大写的参数名称.

# parser.add_argument('--DIR_in', metavar='DIR_FULL_IMAGE_IN', default=r'/root/autodl-tmp/data/target5', type=str,
#                     help='path to input full ele dyna or state dataset')
# parser.add_argument('--DIR_out_train', default=r'/root/autodl-tmp/data/stage_3_dyna_stat/train', type=str, metavar='DIR_SPILT_IMAGE_OUT_PATH',
#                     help='path to out spilted windows ele img')
# parser.add_argument('--DIR_out_val', default=r'/root/autodl-tmp/data/stage_3_dyna_stat/val', type=str, metavar='DIR_SPILT_IMAGE_OUT_PATH',
#                     help='path to out spilted windows ele img')
parser.add_argument('--DIR_in', metavar='DIR_FULL_IMAGE_IN', default=r'D:\Data\target5', type=str,
# parser.add_argument('--DIR_in', metavar='DIR_FULL_IMAGE_IN', default=r'/root/autodl-tmp/data/target5', type=str,
                    help='path to input full ele dyna or state dataset')
parser.add_argument('--DIR_out_train', default=r'D:\Data\target_stage3_small\train', type=str, metavar='DIR_SPILT_IMAGE_OUT_PATH',
                    help='path to out spilted windows ele img')
parser.add_argument('--DIR_out_val', default=r'D:\Data\target_stage3_small\val', type=str, metavar='DIR_SPILT_IMAGE_OUT_PATH',
                    help='path to out spilted windows ele img')

# parser.add_argument('--DIR_unsupervised_in', metavar='DIR_FULL_IMAGE_IN', default=r'/root/autodl-tmp/data/target4', type=str,
#                     help='path to input full ele dyna or state dataset')
# parser.add_argument('--DIR_unsupervised_out', metavar='DIR_FULL_IMAGE_IN', default=r'/root/autodl-tmp/data/target_stage_1_dyna', type=str,
#                     help='path to input full ele dyna or state dataset')
# parser.add_argument('--stat_use', default=True, type=bool, help='whether spilt stat img')
parser.add_argument('--DIR_unsupervised_in', metavar='DIR_FULL_IMAGE_IN', default=r'D:\Data\target4', type=str,
                    help='path to input full ele dyna or state dataset')
parser.add_argument('--DIR_unsupervised_out', metavar='DIR_FULL_IMAGE_IN', default=r'D:\Data\target_stage1_small_big_mix', type=str,
                    help='path to input full ele dyna or state dataset')

parser.add_argument('--windows_length', default=0.625, type=float, help='how long every windows is')
parser.add_argument('--windows_step', default=0.5, type=float, help='how long every step is')
# parser.add_argument('--target_folder_charter', default=['LG7-8', 'LG7-4', 'LG7-16', 'LG701', 'LG701-H1', 'LN11-4', 'LN11-5'],
#                     type=list, help='which folder is going to process')
parser.add_argument('--target_folder_charter', default=['LG7-4', 'LG7-8', 'LG7-11', 'LG7-12', 'LG7-16',
                                                        'LG701', 'LG701-H1'],
                    type=list, help='which folder is going to process')
parser.add_argument('--target_folder_charter_val', default=['LN11-4', 'LN11-5'],
                    type=list, help='which folder is going to process')

args = parser.parse_args()
print(args)

# 根据类别，对大的电成像文件 进行分割 成 小的 ImageFolder 形式
def start_pic_spilt_by_categray_for_supervised_learning_stage_2(args):
    # 遍历访问 输入文件夹下的所有文件夹
    folder_list = traverseFolder_folder(args.DIR_in)

    # 对所有文件夹进行判断，判断是否是 目标文件夹
    target_folder = []
    for i in range(len(folder_list)):
        for j in range(len(args.target_folder_charter)):
            if folder_list[i].__contains__(args.target_folder_charter[j]):
                target_folder.append(folder_list[i])
                break
    print(folder_list)
    print(target_folder)

    # 把每一个文件夹，实例化成一个 dataset 对象, 并进行分割
    dataset_ele = []
    for i in range(len(target_folder)):
        print('loading pic from path:{}'.format(target_folder[i].split('/')[-1]))
        a = Ele_data_img(target_folder[i], args)
        a.pic_auto_spilt_by_layer()
        # dataset_ele.append(a)

    pass


# 无监督数据生成，用于无监督训练，阶段1的图像数据
def start_pic_spilt_directly_for_unsupervised_learning_stage_1(args):
    # 遍历访问 输入文件夹下的所有文件夹
    folder_list = traverseFolder_folder(args.DIR_unsupervised_in)

    target_folder = folder_list

    # 把每一个文件夹，实例化成一个 dataset 对象, 并进行分割
    dataset_ele = []
    print(target_folder)
    for i in range(len(target_folder)):
        print('loading pic from path:{}'.format(target_folder[i].split('/')[-1]))
        a = Ele_data_img(target_folder[i])
        a.pic_spilt_directlly_for_unsupervised_learning(args.DIR_unsupervised_out)
        # dataset_ele.append(a)


# 根据类别，对电成像文件 进行分割 成 大的 层段状文件，每一个层段分成一个问价，ImageFolder 形式
def start_pic_spilt_by_categray_for_supervised_learning_stage_3(args):
    # 遍历访问 输入文件夹下的所有文件夹
    folder_list = traverseFolder_folder(args.DIR_in)

    # 训练集文件生成
    # 对所有文件夹进行判断，判断是否是 目标文件夹
    target_folder = []
    for i in range(len(folder_list)):
        for j in range(len(args.target_folder_charter)):
            if folder_list[i].__contains__(args.target_folder_charter[j]):
                target_folder.append(folder_list[i])
                break
    print(target_folder)
    # print(target_folder)

    # 把每一个文件夹，实例化成一个 dataset 对象, 并进行分割
    dataset_ele = []
    for i in range(len(target_folder)):
        print('loading pic from path:{}'.format(target_folder[i].split('/')[-1]))
        a = Ele_data_img(target_folder[i], args)
        a.pic_auto_spilt_by_layer_to_small_layer(args.DIR_out_train)
        # dataset_ele.append(a)

    # 测试集文件生成
    # 对所有文件夹进行判断，判断是否是 目标文件夹
    target_folder = []
    for i in range(len(folder_list)):
        for j in range(len(args.target_folder_charter_val)):
            if folder_list[i].__contains__(args.target_folder_charter_val[j]):
                target_folder.append(folder_list[i])
                break
    print(target_folder)
    # print(target_folder)

    # 把每一个文件夹，实例化成一个 dataset 对象, 并进行分割
    dataset_ele = []
    for i in range(len(target_folder)):
        print('loading pic from path:{}'.format(target_folder[i].split('/')[-1]))
        a = Ele_data_img(target_folder[i], args)
        a.pic_auto_spilt_by_layer_to_small_layer(args.DIR_out_val)
        # dataset_ele.append(a)

if __name__ == '__main__':
    start_pic_spilt_directly_for_unsupervised_learning_stage_1(args)
    # start_pic_spilt_by_categray_for_supervised_learning_stage_2(args)
    # start_pic_spilt_by_categray_for_supervised_learning_stage_3(args)