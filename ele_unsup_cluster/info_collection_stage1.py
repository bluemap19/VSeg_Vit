import argparse
import os

import numpy as np
import torch
import torchvision.models as models
from tqdm import trange

from simsiam.sim_model import model_stage1
from simsiam.ELE_data import Ele_data_img
from simsiam.sim_model.model_stage1 import SimSiam
from simsiam.sim_model.model_stage_1_vit import vit_simsiam
from src_ele.dir_operation import traverseFolder_folder

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('--DIR_in', metavar='DIR', default=r'D:\Data\target5', type=str,
parser.add_argument('--DIR_in', metavar='DIR', default=r'D:\Data\pic_temp', type=str,
                    help='path to dataset')
# parser.add_argument('--Model_stage1_path', metavar='modle_path', default=r'D:\Data\target_answer\256-dyna-stat\checkpoint_VIT_8patch_56_0014.pth.tar',
parser.add_argument('--Model_stage1_path', metavar='modle_path', default=r'D:\Data\target_answer\250X250\checkpoint_res50_batch240_dim384_epoch0039.pth.tar',
# parser.add_argument('--Model_stage1_path', metavar='modle_path', default=r'D:\Data\target_answer\250X250\checkpoint_res50_batch240_dim15_epoch0016.pth.tar',
                    type=str, help='path to dataset')
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
    # print("=> creating model '{}'".format(args.arch))
    # model = model_simSiam_stage3(class_num=1)
    # model_temp = model_stage1.SimSiam()
    dim, out = 512, 384
    # dim, out = 20, 15
    # model_temp = vit_simsiam()
    model_temp = SimSiam(2, dim, out)

    if os.path.isfile(args.Model_stage1_path):
        print("=> loading checkpoint '{}'".format(args.Model_stage1_path))
        DEVICE = torch.device("cpu")
        model_temp.to(DEVICE)
        checkpoint = torch.load(args.Model_stage1_path, map_location=DEVICE)
        model_temp.load_state_dict(checkpoint['model_dict'])    # model_dict
    else:
        print('no model information')
        exit(0)
    # model_temp = model_temp.arch_model
    model_temp = model_temp.encoder

    all_feature_stage1 = []
    # all_stat_feature_stage1 = []
    depth_info = []
    well_name_info = []

    path_list = traverseFolder_folder(args.DIR_in)
    model_temp.eval()
    for i in path_list:
        print(i)
        ELE_Temp = Ele_data_img(i)
        dep_start = ELE_Temp.dep_start
        dep_end = ELE_Temp.dep_end
        step = 0.2
        windows_length = 0.63
        pic_num = int((dep_end-dep_start-windows_length)/step+1)
        for j in trange(pic_num):
            depth_temp = dep_start + (j+1)*step
            pic_dyna_windows, pic_static_windows, depth_data_windows = ELE_Temp.get_pic_from_depth(depth_temp, thickness=windows_length)
            # pic_dyna_windows = torch.reshape(torch.from_numpy(pic_dyna_windows).float(), (1, 1, 224, 224))
            # pic_static_windows = torch.reshape(torch.from_numpy(pic_static_windows).float(), (1, 1, 224, 224))

            pic_dyna = pic_dyna_windows.reshape((1, 224, 224))
            pic_stat = pic_static_windows.reshape((1, 224, 224))
            pic_all = np.append(pic_dyna, pic_stat, axis=0)

            pic_all = torch.reshape(torch.from_numpy(pic_all).float(), (1, 2, 224, 224))
            r_temp = model_temp(pic_all)

            # print(r_temp['z1'].cpu().detach().numpy().shape)
            # print(pic_all.shape, r_temp.shape)

            all_feature_stage1.append(r_temp.cpu().detach().numpy().ravel())

            depth_info.append([depth_data_windows[0, 0], depth_data_windows[-1, 0]])
            well_name_info.append(i.split('/')[-1].split('\\')[-1])

    all_feature_stage1 = np.array(all_feature_stage1)
    # all_stat_feature_stage1 = np.array(all_stat_feature_stage1)
    depth_info = np.array(depth_info)

    np.savetxt('SeTan_all_feature_{}_res50.txt'.format(dim), all_feature_stage1, delimiter='\t', comments='', fmt='%.4f')
    # np.savetxt('stage1_stat_feature.txt', all_stat_feature_stage1, delimiter='\t', comments='', fmt='%.4f')
    np.savetxt('SeTan_depth_info.txt', depth_info, delimiter='\t', comments='', fmt='%.4f')
    # np.savetxt('stage1_well_name.txt', well_name_info, delimiter='\t', comments='', fmt='%s')

    # print(model_temp)




if __name__ == '__main__':
    stage1_info_coll()