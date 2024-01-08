import argparse
import time

import cv2
import numpy as np
import torch
from torch import nn
from tqdm import trange

from ele_unsup_cluster.get_pca_model_ele import get_pca_model_ele_info
from simsiam.ELE_data import Ele_data_img
from simsiam.sim_model.model_stage1 import SimSiam
from simsiam.sim_model.model_stage3 import model_simSiam_stage3, Model_simple_linear
from src_ele.dir_operation import traverseFolder_folder, traverseFolder, check_and_make_dir
from src_ele.file_operation import get_ele_data_from_path
from src_ele.pic_opeeration import get_pic_distribute


class CombinedModel(nn.Module):
    def __init__(self, submodel1, submodel2):
        super(CombinedModel, self).__init__()
        self.submodel1 = submodel1
        self.submodel2 = submodel2

    def forward(self, x):
        x = self.submodel1(x)
        # print(x.shape)
        x = self.submodel2(x)
        return x


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# metavar - 在 usage 说明中的参数名称，对于必选参数默认就是参数名称，对于可选参数默认是全大写的参数名称.


parser.add_argument('--DIR_in', metavar='DIR_FULL_IMAGE_IN', default=r'D:\Data\target5', type=str,
                    help='path to input full ele dyna or state dataset')
parser.add_argument('--windows_length', default=250*0.0025, type=float, help='how long every windows is')
parser.add_argument('--windows_step', default=0.01, type=float, help='how long every step is')
# parser.add_argument('--target_folder_charter', default=['LG7-8', 'LG7-4', 'LG7-16', 'LG701', 'LG701-H1', 'LN11-4', 'LN11-5'],
#                     type=list, help='which folder is going to process')
parser.add_argument('--target_folder_charter', default=['LG7-4', 'LG7-8', 'LG7-11', 'LG7-12', 'LG7-16',
                                                        'LG701', 'LG701-H1', 'LN11-4', 'LN11-5'],
                    type=list, help='which folder is going to process')
parser.add_argument('--out_dim', default=512, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--in_dim', default=2, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--pred_dim', default=384, type=int,
                    help='hidden dimension of the predictor (default: 512)')
parser.add_argument('--resume_encoder', default=r'D:\Data\target_answer\250X250\checkpoint_res50_batch240_dim384_epoch0039.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume_pred', default=r'D:/GitHubProj/dino/simsiam/checkpoint_predictor_batch64_dim3_epoch0010_b.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--PCA_PATH', default=r'D:\GitHubProj\dino\ele_unsup_cluster\stage1_all_feature_512_res50.txt', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
args = parser.parse_args()
# print(args)


########### ResNet50 主干模型载入
model_pre = SimSiam(args.in_dim, args.out_dim, args.pred_dim)
prev_dim = model_pre.encoder.fc[6].out_features
###### print(prev_dim)
# 预测头模型载入
# model = model_simSiam_stage3(class_num=3, in_dim=16)
# model = Model_simple_linear(in_dim=16, out_dim=3)
model = Model_simple_linear(in_dim=16, out_dim=3)
# 先把主干模型进行冻结
for name, param in model_pre.named_parameters():
    # print(name)
    param.requires_grad = False

DEVICE = torch.device("cpu")
print('CPU processing:{}'.format(DEVICE))
checkpoint = torch.load(args.resume_encoder, map_location=DEVICE)
model_pre.to(DEVICE)
model_pre.load_state_dict(checkpoint['model_dict'])  # model_dict
model_pre = model_pre.encoder


checkpoint = torch.load(args.resume_pred, map_location=DEVICE)
model.to(DEVICE)
model.load_state_dict(checkpoint['state_dict'])  # model_dict

PCA = get_pca_model_ele_info(path=args.PCA_PATH, dim=6)
# model_pre = SimSiam(args.in_dim, args.out_dim, args.pred_dim)
# prev_dim = model_pre.encoder.fc[6].out_features
# model = model_simSiam_stage3(class_num=4, in_dim=prev_dim)
# model_pre = model_pre.encoder
# model_final = CombinedModel(model_pre, model)
#
# DEVICE = torch.device("cpu")
# print('CPU processing:{}'.format(DEVICE))
#
# checkpoint = torch.load(args.resume_encoder, map_location=DEVICE)
# model_final.load_state_dict(checkpoint['model_dict'])    # model_dict
# model_final.to(DEVICE)
# model_final.eval()


# # 根据文件夹，读取成像测井数据，对成像测井进行储层类别预测，并把预测结果和深度相结合进行保存
def layer_pred(args):
    # 遍历访问 输入文件夹下的所有文件夹
    folder_list = traverseFolder_folder(args.DIR_in)

    # 对所有文件夹进行判断，判断是否是 目标文件夹
    target_folder = []
    for i in range(len(folder_list)):
        for j in range(len(args.target_folder_charter)):
            if folder_list[i].__contains__(args.target_folder_charter[j]):
                target_folder.append(folder_list[i])
                break
    # print(folder_list)
    print(target_folder)

    # # 把每一个文件夹，实例化成一个 dataset 对象, 并进行分割
    # dataset_ele = []
    for i in range(len(target_folder)):
        print('loading pic from path:{}'.format(target_folder[i].split('/')[-1]))
        a = Ele_data_img(target_folder[i], args)
        num_split = int((a.dep_end - a.dep_start - args.windows_length)/args.windows_step)
        pic_loader_my = []
        color_feature_loader_my = []
        answer = np.zeros((num_split, 4))
        time_t = time.time()

        for j in trange(num_split):
            dep_tmp = j*args.windows_step + a.dep_start + args.windows_length/2
            answer[j, 0] = dep_tmp
            # print(dep_tmp, a.dep_end , a.dep_start)
            pic_dyna_windows, pic_static_windows, depth_data_windows = a.get_pic_from_depth(dep_tmp, args.windows_length)
            pic_loader_my.append(np.array([pic_dyna_windows, pic_static_windows]))

            print('d1 {}'.format(time.time() - time_t))
            time_t = time.time()
            color_feature = get_pic_distribute(pic_static_windows, dist_length=9, min_V=0, max_V=1)
            print('d2 {}'.format(time.time() - time_t))
            time_t = time.time()
            color_feature = np.append(color_feature, np.mean(pic_static_windows))
            color_feature_loader_my.append(color_feature)

            if ((j+1)%20==0):
                pic_loader_my = np.array(pic_loader_my).reshape((20, 2, 224, 224))
                pic_temp = torch.from_numpy(pic_loader_my)
                print('d3 {}'.format(time.time() - time_t))
                time_t = time.time()
                pic_feature = model_pre(pic_temp).cpu().detach().numpy()
                print('d4 {}'.format(time.time() - time_t))
                time_t = time.time()
                pic_feature_PCA = PCA.transform(pic_feature)
                print('f5 {}'.format(time.time() - time_t))
                time_t = time.time()

                print(np.array(pic_loader_my).shape, np.array(color_feature_loader_my).shape, pic_feature_PCA.shape)

                color_feature = np.array(color_feature_loader_my)

                pred_feature_in = np.append(pic_feature_PCA, color_feature, axis=1)
                pred_feature_in = torch.from_numpy(pred_feature_in).float()
                print('d6 {}'.format(time.time() - time_t))
                time_t = time.time()
                pred_answer = model(pred_feature_in).cpu().detach().numpy()
                print('d7 {}'.format(time.time() - time_t))
                time_t = time.time()

                answer[j-19:j+1, 1:] = pred_answer

                pic_loader_my = []
                color_feature_loader_my = []
                pass

            # print(pic_temp.shape)
            # print(pic_feature.shape, pic_feature_PCA.shape)
            # answer_temp[1:] = pred_answer.cpu().detach().numpy()
            # # print(dep_tmp, answer_temp)
            # answer.append(answer_temp)
            # # print(answer_temp.cpu().detach().numpy())
            # # exit(0)

        answer = np.array(answer)
        for j in range(answer.shape[0]):
            for k in range(3):
                if answer[j][k+1] < 0:
                    answer[j][k+1] = 0
                if answer[j][k+1] > 1:
                    answer[j][k+1] = 1
        np.savetxt('{}/{}_pred_answer.txt'.format(target_folder[i], target_folder[i].split('/')[-1]), answer, fmt='%.4f',
                   delimiter='\t', newline='\n', header='', comments='')
#     #     a.pic_auto_spilt_by_layer()
#     #     # dataset_ele.append(a)



if __name__ == '__main__':
    layer_pred(args)
    pass