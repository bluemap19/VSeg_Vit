import argparse
import os

import numpy as np
import torch
import torchvision.models as models
from tqdm import trange

from simsiam.sim_model import model_stage1
from simsiam.ELE_data import Ele_data
from src_ele.dir_operation import traverseFolder_folder, check_and_make_dir

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))



parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--DIR_in', metavar='DIR', default=r'D:\Data\target5', type=str,
                    help='path to dataset')
parser.add_argument('--DIR_out', metavar='DIR', default=r'D:\Data\target_uns_class', type=str,
                    help='path to dataset')
parser.add_argument('--Model_stage1_path', metavar='modle_path', default=r'D:\PycharmProjects\file\checkpoint_0148_dyst_0137_dyna_256.pth.tar', type=str,
                    help='path to dataset')
parser.add_argument('--stat_use', metavar='STAT_USE', default=True, type=bool,
                    help='whether use stat img')
########################################################################################
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
###############################################################################

# 该函数主要进行无监督聚类   输入数据的收集
# 输入数据来自于stage1模型的输出
def stage1_info_coll():
    args = parser.parse_args()

    # print model name
    print("=> creating model '{}'".format(args.arch))
    # model = model_simSiam_stage3(class_num=1)
    model_temp = model_stage1.SimSiam(models.__dict__[args.arch])

    for i in range(10):
        check_and_make_dir(args.DIR_out+'\\{}'.format(i))


    if os.path.isfile(args.Model_stage1_path):
        print("=> loading checkpoint '{}'".format(args.Model_stage1_path))
        DEVICE = torch.device("cpu")
        model_temp.to(DEVICE)
        checkpoint = torch.load(args.Model_stage1_path, map_location=DEVICE)
        model_temp.load_state_dict(checkpoint['model_dict'])    # model_dict

    all_dyna_feature_stage1 = []
    all_stat_feature_stage1 = []
    depth_info = []
    well_name_info = []
    class_info = np.loadtxt('0.5step-10class/all_pred.txt', delimiter='\t', encoding='GBK')
    print(class_info.shape)

    path_list = traverseFolder_folder(args.DIR_in)
    model_temp.eval()
    K = 0
    for i in path_list:
        print(i)
        ELE_Temp = Ele_data(i, args)
        dep_start = ELE_Temp.dep_start
        dep_end = ELE_Temp.dep_end
        step = 0.5
        windows_length = 2
        pic_num = int((dep_end-dep_start-windows_length)/step+1)
        for j in trange(pic_num):
            # depth_temp = dep_start + (j+1)*step
            depth_temp = class_info[K, 0]
            DIR_Temp = args.DIR_out + '\\{}'.format(int(class_info[K, -1]))
            # print(DIR_Temp)
            pic_dyna_windows, pic_static_windows, depth_data_windows = ELE_Temp.get_pic_from_depth(depth_temp, save=True, DIR=DIR_Temp)
            K = K + 1
    #         pic_dyna_windows = torch.reshape(torch.from_numpy(pic_dyna_windows).float(), (1, 1, 196, 196))
    #         pic_static_windows = torch.reshape(torch.from_numpy(pic_static_windows).float(), (1, 1, 196, 196))
    #
    #         # print(pic_dyna_windows.shape, pic_static_windows.shape)
    #
    #         r_temp = model_temp(pic_dyna_windows, pic_static_windows)
    #
    #         # print(r_temp['z1'].cpu().detach().numpy().shape)
    #
    #         all_dyna_feature_stage1.append(r_temp['z1'].cpu().detach().numpy().ravel())
    #         all_stat_feature_stage1.append(r_temp['z2'].cpu().detach().numpy().ravel())
    #
    #         depth_info.append([depth_data_windows[0, 0], depth_data_windows[-1, 0]])
    #         well_name_info.append(i.split('/')[-1].split('\\')[-1])
    #
    # all_dyna_feature_stage1 = np.array(all_dyna_feature_stage1)
    # all_stat_feature_stage1 = np.array(all_stat_feature_stage1)
    # depth_info = np.array(depth_info)
    #
    # np.savetxt('stage1_dyna_feature.txt', all_dyna_feature_stage1, delimiter='\t', comments='', fmt='%.4f')
    # np.savetxt('stage1_stat_feature.txt', all_stat_feature_stage1, delimiter='\t', comments='', fmt='%.4f')
    # np.savetxt('stage1_depth_info.txt', depth_info, delimiter='\t', comments='', fmt='%.4f')
    # np.savetxt('stage1_well_name.txt', well_name_info, delimiter='\t', comments='', fmt='%s')

    # print(model_temp)

if __name__ == '__main__':
    stage1_info_coll()